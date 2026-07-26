import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import '../config/app_config.dart';
import '../services/ad_consent_service.dart';
import '../services/native_platform_bridge.dart';

class NativeMoreAd extends StatefulWidget {
  const NativeMoreAd({super.key});

  @override
  State<NativeMoreAd> createState() => _NativeMoreAdState();
}

class _NativeMoreAdState extends State<NativeMoreAd>
    with WidgetsBindingObserver {
  static const _retryDelays = [
    Duration(seconds: 2),
    Duration(seconds: 5),
    Duration(seconds: 10),
    Duration(seconds: 30),
  ];

  NativeAd? _ad;
  bool _loaded = false;
  bool _loading = false;
  Timer? _retryTimer;
  int _retryAttempt = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    AdConsentService.instance.canRequestAds.addListener(_onConsentChanged);
    if (AppConfig.adsEnabled && AdConsentService.instance.canRequestAds.value) {
      _load();
    }
  }

  void _onConsentChanged() {
    if (AdConsentService.instance.canRequestAds.value && !_loaded) {
      _retryNow();
    }
    if (mounted) setState(() {});
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && !_loaded) {
      _retryNow();
    }
  }

  void _retryNow() {
    _retryTimer?.cancel();
    _retryTimer = null;
    _load();
  }

  void _load() {
    if (!AppConfig.adsEnabled ||
        _loading ||
        _loaded ||
        !AdConsentService.instance.canRequestAds.value) {
      return;
    }
    final unitId = NativePlatformBridge.runtimeConfig?.nativeMoreAdUnitId ?? '';
    if (unitId.isEmpty) return;
    _loading = true;
    late final NativeAd ad;
    ad = NativeAd(
      adUnitId: unitId,
      request: const AdRequest(),
      listener: NativeAdListener(
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
      factoryId: defaultTargetPlatform == TargetPlatform.android
          ? 'moreNativeAd'
          : null,
      nativeTemplateStyle: defaultTargetPlatform == TargetPlatform.iOS
          ? NativeTemplateStyle(
              templateType: TemplateType.medium,
              mainBackgroundColor: Colors.white,
              cornerRadius: 16,
              callToActionTextStyle: NativeTemplateTextStyle(
                textColor: Colors.white,
                backgroundColor: Color(0xFF0D6E6E),
                style: NativeTemplateFontStyle.bold,
                size: 15,
              ),
              primaryTextStyle: NativeTemplateTextStyle(
                textColor: Color(0xFF101820),
                style: NativeTemplateFontStyle.bold,
                size: 16,
              ),
              secondaryTextStyle: NativeTemplateTextStyle(
                textColor: Color(0xFF66746F),
                style: NativeTemplateFontStyle.normal,
                size: 13,
              ),
              tertiaryTextStyle: NativeTemplateTextStyle(
                textColor: Color(0xFF66746F),
                style: NativeTemplateFontStyle.normal,
                size: 12,
              ),
            )
          : null,
    );
    _ad = ad;
    unawaited(
      ad.load().catchError((Object _) {
        if (identical(_ad, ad)) {
          _ad = null;
          _loading = false;
          ad.dispose();
          if (mounted) setState(() => _loaded = false);
          _scheduleRetry();
        }
      }),
    );
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
      if (mounted) _load();
    });
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
  Widget build(BuildContext context) {
    if (!AdConsentService.instance.canRequestAds.value) {
      return const SizedBox.shrink();
    }
    final ad = _ad;
    if (!_loaded || ad == null) return const SizedBox.shrink();
    return SizedBox(height: 350, child: AdWidget(ad: ad));
  }
}
