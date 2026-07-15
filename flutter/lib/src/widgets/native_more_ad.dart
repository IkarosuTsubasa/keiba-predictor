import 'package:flutter/material.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import '../config/app_config.dart';
import '../services/android_platform_bridge.dart';

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
    if (AppConfig.adsEnabled) _load();
  }

  void _load() {
    final unitId =
        AndroidPlatformBridge.runtimeConfig?.nativeMoreAdUnitId ?? '';
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
          if (mounted) setState(() => _loaded = false);
        },
      ),
      factoryId: 'moreNativeAd',
    )..load();
  }

  @override
  void dispose() {
    _ad?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ad = _ad;
    if (!_loaded || ad == null) return const SizedBox.shrink();
    return SizedBox(height: 350, child: AdWidget(ad: ad));
  }
}
