import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../config/app_config.dart';
import '../services/push_notification_service.dart';
import '../web/app_web.dart';
import '../web/web_content_view.dart';
import '../widgets/banner_ad_slot.dart';
import '../widgets/native_more_ad.dart';
import '../widgets/native_nav_icon.dart';
import 'custom_prediction_screen.dart';
import 'more_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  static const _webDestinationCount = 3;
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
      icon: NativeNavIcon(NativeNavIconKind.reports),
      selectedIcon: NativeNavIcon(NativeNavIconKind.reports),
      label: '日報',
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
  late final Uri _baseUri;
  late AppPageChrome _webChrome;
  int _selectedIndex = 0;
  int _lastWebIndex = 0;
  bool _customPredictionActivated = false;
  StreamSubscription<NotificationDestination>? _notificationSubscription;

  @override
  void initState() {
    super.initState();
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
    _notificationSubscription?.cancel();
    super.dispose();
  }

  Future<void> _openNotification(NotificationDestination destination) async {
    final rawUrl = destination.url;
    if (rawUrl != null) {
      final uri = AppWeb.normalizeInAppUrl(rawUrl, AppConfig.baseWebUrl);
      if (uri != null) {
        final index = AppWeb.sectionFor(uri, _baseUri).index;
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
      'reports' => 2,
      'customPrediction' ||
      'custom_prediction' ||
      'my_prediction' => _customPredictionIndex,
      'more' => _moreIndex,
      _ => 0,
    };
    await _selectDestination(index);
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
        _selectedIndex = index;
      });
      return;
    }

    final section = AppSection.values[index];
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
    final sectionIndex = section.index;
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
                  _customPredictionIndex => 1,
                  _moreIndex => 2,
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
                  ),
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
