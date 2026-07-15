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

class _NativeMoreAdState extends State<NativeMoreAd> {
  NativeAd? _ad;
  bool _loaded = false;

  @override
  void initState() {
    super.initState();
    AdConsentService.instance.canRequestAds.addListener(_onConsentChanged);
    if (AppConfig.adsEnabled && AdConsentService.instance.canRequestAds.value) {
      _load();
    }
  }

  void _onConsentChanged() {
    if (AdConsentService.instance.canRequestAds.value && _ad == null) {
      _load();
    }
    if (mounted) setState(() {});
  }

  void _load() {
    if (_ad != null || !AdConsentService.instance.canRequestAds.value) return;
    final unitId = NativePlatformBridge.runtimeConfig?.nativeMoreAdUnitId ?? '';
    if (unitId.isEmpty) return;
    _ad = NativeAd(
      adUnitId: unitId,
      request: const AdRequest(),
      listener: NativeAdListener(
        onAdLoaded: (_) {
          if (mounted) setState(() => _loaded = true);
        },
        onAdFailedToLoad: (ad, _) {
          ad.dispose();
          if (identical(_ad, ad)) _ad = null;
          if (mounted) setState(() => _loaded = false);
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
    )..load();
  }

  @override
  void dispose() {
    AdConsentService.instance.canRequestAds.removeListener(_onConsentChanged);
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
