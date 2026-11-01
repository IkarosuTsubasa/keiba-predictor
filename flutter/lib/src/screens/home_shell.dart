import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../config/app_config.dart';
import '../services/ad_consent_service.dart';
import '../services/push_notification_service.dart';
import '../web/app_web.dart';
import '../web/web_content_view.dart';
import '../widgets/banner_ad_slot.dart';
import '../widgets/native_more_ad.dart';
import '../widgets/native_nav_icon.dart';
import 'custom_prediction_screen.dart';
import 'more_screen.dart';
import 'simulation_home_screen.dart';
import '../models/simulation_context.dart';
import '../services/simulation_analytics.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => HomeShellState();
}

class HomeShellState extends State<HomeShell> with WidgetsBindingObserver {
  static const _webDestinationCount = 2;
  static const _simulationIndex = 2;
  static const _customPredictionIndex = 3;
  static const _moreIndex = 4;

  static const _destinations = [
    NavigationDestination(
      icon: NativeNavIcon(NativeNavIconKind.races),
      selectedIcon: NativeNavIcon(NativeNavIconKind.races),
      label: 'レース',
    ),
    NavigationDestination(
      icon: NativeNavIcon(NativeNavIconKind.history),
      selectedIcon: NativeNavIcon(NativeNavIconKind.history),
      label: '履歴分析',
    ),
    NavigationDestination(
      icon: NativeNavIcon(NativeNavIconKind.simulation),
      selectedIcon: NativeNavIcon(NativeNavIconKind.simulation),
      label: 'シミュレーション',
    ),
    NavigationDestination(
      icon: NativeNavIcon(NativeNavIconKind.customPrediction),
      selectedIcon: NativeNavIcon(NativeNavIconKind.customPrediction),
      label: 'My予測',
    ),
    NavigationDestination(
      icon: NativeNavIcon(NativeNavIconKind.more),
      selectedIcon: NativeNavIcon(NativeNavIconKind.more),
      label: 'その他',
    ),
  ];

  static const _moreChrome = AppPageChrome(
    title: 'その他',
    subtitle: 'ご利用案内とお問い合わせを確認できます',
  );

  final _webKey = GlobalKey<WebContentViewState>();
  final _simulationKey = GlobalKey<SimulationHomeScreenState>();
  bool _simulationActivated = false;
  late final Uri _baseUri;
  late AppPageChrome _webChrome;
  int _selectedIndex = 0;
  int _lastWebIndex = 0;
  bool _customPredictionActivated = false;
  StreamSubscription<NotificationDestination>? _notificationSubscription;
  Future<void>? _networkRecoveryInProgress;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _baseUri = Uri.parse(AppConfig.baseWebUrl);
    _webChrome = AppWeb.chromeFor(
      AppWeb.topLevelUri(AppConfig.baseWebUrl, AppSection.races),
      _baseUri,
    );
    if (AppConfig.notificationsEnabled) {
      _notificationSubscription = PushNotificationService.instance.destinations
          .listen((destination) => unawaited(_openNotification(destination)));
      WidgetsBinding.instance.addPostFrameCallback((_) {
        final pending = PushNotificationService.instance
            .takePendingDestination();
        if (pending != null) {
          unawaited(_openNotification(pending));
        }
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _notificationSubscription?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        unawaited(_recoverAfterNetworkPermission());
      }
    });
  }

  Future<void> _recoverAfterNetworkPermission() {
    final existing = _networkRecoveryInProgress;
    if (existing != null) return existing;

    late final Future<void> operation;
    operation = _performNetworkRecovery().whenComplete(() {
      if (identical(_networkRecoveryInProgress, operation)) {
        _networkRecoveryInProgress = null;
      }
    });
    _networkRecoveryInProgress = operation;
    return operation;
  }

  Future<void> _performNetworkRecovery() async {
    final web = _webKey.currentState;
    try {
      await Future.wait([
        AdConsentService.instance.retryAfterNetworkRecovery(),
        if (web != null) web.retryAfterNetworkInterruption(),
      ]);
    } catch (error) {
      debugPrint('Network recovery retry failed: $error');
    }
  }

  Future<void> _openNotification(NotificationDestination destination) async {
    final rawUrl = destination.url;
    if (rawUrl != null) {
      final uri = AppWeb.normalizeInAppUrl(rawUrl, AppConfig.baseWebUrl);
      if (uri != null) {
        if (_handleSimulationUri(uri)) return;
        final section = AppWeb.sectionFor(uri, _baseUri);
        if (section == AppSection.reports) {
          _openLegacyReport(uri);
          return;
        }
        final index = section == AppSection.history ? 1 : 0;
        if (mounted) {
          setState(() {
            _selectedIndex = index;
            _lastWebIndex = index;
            _webChrome = AppWeb.chromeFor(uri, _baseUri);
          });
        }
        await _webKey.currentState?.load(uri);
        return;
      }
    }

    final index = switch (destination.destination) {
      'history' => 1,
      'simulation' || 'simulator' => _simulationIndex,
      'customPrediction' ||
      'custom_prediction' ||
      'my_prediction' => _customPredictionIndex,
      'more' => _moreIndex,
      _ => 0,
    };
    if (destination.destination == 'reports') {
      _openLegacyReport(
        AppWeb.topLevelUri(AppConfig.baseWebUrl, AppSection.reports),
      );
      return;
    }
    await _selectDestination(index);
  }

  void _openLegacyReport(Uri uri) {
    if (!mounted) return;
    unawaited(
      Navigator.of(context).push<void>(
        MaterialPageRoute(
          builder: (context) => Scaffold(
            appBar: AppBar(title: const Text('私の日報')),
            body: WebContentView(initialUri: uri, baseUri: _baseUri),
          ),
        ),
      ),
    );
  }

  bool _handleSimulationUri(Uri uri) {
    final simulation = SimulationContext.fromUri(uri, _baseUri);
    if (simulation == null) return false;
    openSimulation(simulation);
    return true;
  }

  void openSimulation(SimulationContext simulation) {
    if (!mounted) return;
    setState(() {
      _simulationActivated = true;
      _selectedIndex = _simulationIndex;
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _simulationKey.currentState?.open(simulation);
    });
  }

  Future<void> _selectDestination(int index) async {
    if (index == _selectedIndex) {
      if (index < _webDestinationCount) {
        await _webKey.currentState?.scrollToTop();
      }
      return;
    }

    if (index >= _webDestinationCount) {
      setState(() {
        if (_selectedIndex < _webDestinationCount) {
          _lastWebIndex = _selectedIndex;
        }
        if (index == _customPredictionIndex) {
          _customPredictionActivated = true;
        }
        if (index == _simulationIndex) _simulationActivated = true;
        _selectedIndex = index;
      });
      if (index == _simulationIndex) {
        unawaited(simulationEvent('sim_tab_open', {'source': 'app_tab'}));
      }
      return;
    }

    final section = index == 1 ? AppSection.history : AppSection.races;
    setState(() {
      _selectedIndex = index;
      _lastWebIndex = index;
      _webChrome = AppWeb.chromeFor(
        AppWeb.topLevelUri(AppConfig.baseWebUrl, section),
        _baseUri,
      );
    });
    await _webKey.currentState?.load(
      AppWeb.topLevelUri(AppConfig.baseWebUrl, section),
    );
  }

  void _onLocationChanged(Uri uri) {
    final section = AppWeb.sectionFor(uri, _baseUri);
    final sectionIndex = section == AppSection.history ? 1 : 0;
    if (!mounted) return;
    setState(() {
      _webChrome = AppWeb.chromeFor(uri, _baseUri);
      if (_selectedIndex < _webDestinationCount) {
        _selectedIndex = sectionIndex;
        _lastWebIndex = sectionIndex;
      }
    });
  }

  Future<void> _handleBack() async {
    if (_selectedIndex == _simulationIndex &&
        await _simulationKey.currentState?.handleBack() == true) {
      return;
    }
    if (_selectedIndex >= _webDestinationCount) {
      setState(() => _selectedIndex = _lastWebIndex);
      return;
    }
    final web = _webKey.currentState;
    if (web != null && await web.canGoBack()) {
      await web.goBack();
      return;
    }
    await SystemNavigator.pop();
  }

  @override
  Widget build(BuildContext context) {
    final chrome = switch (_selectedIndex) {
      _simulationIndex => const AppPageChrome(
        title: 'シミュレーション',
        subtitle: 'レースを選んで、展開を試せます',
      ),
      _customPredictionIndex => null,
      _moreIndex => _moreChrome,
      _ => _webChrome,
    };
    final headerColor =
        Theme.of(context).appBarTheme.backgroundColor ??
        const Color(0xFF07111F);
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) {
          unawaited(_handleBack());
        }
      },
      child: Scaffold(
        appBar: chrome == null
            ? null
            : AppBar(
                toolbarHeight: 86,
                titleSpacing: 16,
                title: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      chrome.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 23,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      chrome.subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Color(0xFFC9D4DE),
                        fontSize: 13,
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
        body: Column(
          children: [
            Expanded(
              child: IndexedStack(
                index: switch (_selectedIndex) {
                  _simulationIndex => 1,
                  _customPredictionIndex => 2,
                  _moreIndex => 3,
                  _ => 0,
                },
                children: [
                  WebContentView(
                    key: _webKey,
                    initialUri: AppWeb.topLevelUri(
                      AppConfig.baseWebUrl,
                      AppSection.races,
                    ),
                    baseUri: _baseUri,
                    onLocationChanged: _onLocationChanged,
                    onNavigationRequested: _handleSimulationUri,
                  ),
                  _simulationActivated
                      ? SimulationHomeScreen(
                          key: _simulationKey,
                          active: _selectedIndex == _simulationIndex,
                        )
                      : const SizedBox.shrink(),
                  _customPredictionActivated
                      ? AnnotatedRegion<SystemUiOverlayStyle>(
                          value: const SystemUiOverlayStyle(
                            statusBarColor: Colors.transparent,
                            statusBarIconBrightness: Brightness.light,
                            statusBarBrightness: Brightness.dark,
                          ),
                          child: ColoredBox(
                            color: headerColor,
                            child: SafeArea(
                              bottom: false,
                              child: ColoredBox(
                                color: Theme.of(
                                  context,
                                ).scaffoldBackgroundColor,
                                child: const CustomPredictionScreen(),
                              ),
                            ),
                          ),
                        )
                      : const SizedBox.shrink(),
                  const MoreScreen(),
                ],
              ),
            ),
            if (_selectedIndex < _webDestinationCount) const BannerAdSlot(),
            if (_selectedIndex == _moreIndex)
              const Padding(
                padding: EdgeInsets.fromLTRB(12, 10, 12, 8),
                child: NativeMoreAd(),
              ),
          ],
        ),
        bottomNavigationBar: NavigationBar(
          height: 86,
          selectedIndex: _selectedIndex,
          onDestinationSelected: (index) =>
              unawaited(_selectDestination(index)),
          destinations: _destinations,
        ),
      ),
    );
  }
}
