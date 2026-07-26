import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import '../config/app_config.dart';
import '../services/ad_consent_service.dart';
import '../services/native_platform_bridge.dart';

class BannerAdSlot extends StatefulWidget {
  const BannerAdSlot({super.key});

  @override
  State<BannerAdSlot> createState() => _BannerAdSlotState();
}

class _BannerAdSlotState extends State<BannerAdSlot>
    with WidgetsBindingObserver {
  static const _retryDelays = [
    Duration(seconds: 2),
    Duration(seconds: 5),
    Duration(seconds: 10),
    Duration(seconds: 30),
  ];

  BannerAd? _ad;
  int? _requestedWidth;
  int? _availableWidth;
  bool _loaded = false;
  bool _loading = false;
  Timer? _retryTimer;
  int _retryAttempt = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    AdConsentService.instance.canRequestAds.addListener(_onConsentChanged);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    AdConsentService.instance.canRequestAds.removeListener(_onConsentChanged);
    _retryTimer?.cancel();
    _ad?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && !_loaded) {
      _retryNow();
    }
  }

  void _onConsentChanged() {
    if (AdConsentService.instance.canRequestAds.value && !_loaded) {
      _retryNow();
    }
  }

  void _retryNow() {
    _retryTimer?.cancel();
    _retryTimer = null;
    final width = _availableWidth;
    if (width != null) {
      unawaited(_load(width, force: true));
    }
  }

  Future<void> _load(int width, {bool force = false}) async {
    if (!AppConfig.adsEnabled ||
        !AdConsentService.instance.canRequestAds.value ||
        width <= 0 ||
        _loading ||
        (_loaded && _requestedWidth == width) ||
        (!force && _retryTimer != null)) {
      return;
    }
    _retryTimer?.cancel();
    _retryTimer = null;
    _requestedWidth = width;
    _loading = true;
    final unitId = NativePlatformBridge.runtimeConfig?.bannerAdUnitId ?? '';
    if (unitId.isEmpty) {
      _loading = false;
      return;
    }

    // 現行 Android 版と同じ、画面幅に合わせるアンカー型バナーを使用する。
    // ignore: deprecated_member_use
    final size = await AdSize.getCurrentOrientationAnchoredAdaptiveBannerAdSize(
      width,
    );
    if (size == null || !mounted) {
      _loading = false;
      _scheduleRetry();
      return;
    }

    final previousAd = _ad;
    _ad = null;
    await previousAd?.dispose();
    if (!mounted) {
      _loading = false;
      return;
    }
    late final BannerAd ad;
    ad = BannerAd(
      adUnitId: unitId,
      size: size,
      request: const AdRequest(),
      listener: BannerAdListener(
        onAdLoaded: (_) {
          if (!identical(_ad, ad)) {
            ad.dispose();
            return;
          }
          _loading = false;
          _retryAttempt = 0;
          if (mounted) setState(() => _loaded = true);
        },
        onAdFailedToLoad: (failedAd, _) {
          failedAd.dispose();
          if (identical(_ad, failedAd)) {
            _ad = null;
            _loading = false;
            if (mounted) setState(() => _loaded = false);
            _scheduleRetry();
          }
        },
      ),
    );
    _ad = ad;
    _loaded = false;
    try {
      await ad.load();
    } catch (_) {
      if (identical(_ad, ad)) {
        _ad = null;
        _loading = false;
        await ad.dispose();
        if (mounted) setState(() => _loaded = false);
        _scheduleRetry();
      }
    }
  }

  void _scheduleRetry() {
    if (!mounted ||
        _retryTimer != null ||
        !AdConsentService.instance.canRequestAds.value) {
      return;
    }
    final retryIndex = _retryAttempt < _retryDelays.length
        ? _retryAttempt
        : _retryDelays.length - 1;
    final delay = _retryDelays[retryIndex];
    _retryAttempt += 1;
    _retryTimer = Timer(delay, () {
      _retryTimer = null;
      final width = _availableWidth;
      if (mounted && width != null) {
        unawaited(_load(width, force: true));
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: AdConsentService.instance.canRequestAds,
      builder: (context, canRequestAds, _) {
        if (!AppConfig.adsEnabled || !canRequestAds) {
          return const SizedBox.shrink();
        }
        return LayoutBuilder(
          builder: (context, constraints) {
            final width = constraints.maxWidth.truncate();
            _availableWidth = width;
            if (width > 0 && !_loading && !_loaded && _retryTimer == null) {
              WidgetsBinding.instance.addPostFrameCallback((_) => _load(width));
            } else if (width > 0 && _loaded && width != _requestedWidth) {
              WidgetsBinding.instance.addPostFrameCallback(
                (_) => _load(width, force: true),
              );
            }
            final ad = _ad;
            if (!_loaded || ad == null) return const SizedBox.shrink();
            return ColoredBox(
              color: Colors.white,
              child: Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Center(
                  child: SizedBox(
                    width: ad.size.width.toDouble(),
                    height: ad.size.height.toDouble(),
                    child: AdWidget(ad: ad),
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }
}
