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

class _BannerAdSlotState extends State<BannerAdSlot> {
  BannerAd? _ad;
  int? _requestedWidth;
  bool _loaded = false;

  @override
  void dispose() {
    _ad?.dispose();
    super.dispose();
  }

  Future<void> _load(int width) async {
    if (!AppConfig.adsEnabled ||
        !AdConsentService.instance.canRequestAds.value ||
        width <= 0 ||
        _requestedWidth == width) {
      return;
    }
    _requestedWidth = width;
    final unitId = NativePlatformBridge.runtimeConfig?.bannerAdUnitId ?? '';
    if (unitId.isEmpty) return;

    // 現行 Android 版と同じ、画面幅に合わせるアンカー型バナーを使用する。
    // ignore: deprecated_member_use
    final size = await AdSize.getCurrentOrientationAnchoredAdaptiveBannerAdSize(
      width,
    );
    if (size == null || !mounted) return;

    await _ad?.dispose();
    final ad = BannerAd(
      adUnitId: unitId,
      size: size,
      request: const AdRequest(),
      listener: BannerAdListener(
        onAdLoaded: (_) {
          if (mounted) setState(() => _loaded = true);
        },
        onAdFailedToLoad: (failedAd, _) {
          failedAd.dispose();
          if (identical(_ad, failedAd)) _ad = null;
          if (mounted) setState(() => _loaded = false);
        },
      ),
    );
    _ad = ad;
    _loaded = false;
    await ad.load();
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
            if (width > 0 && width != _requestedWidth) {
              WidgetsBinding.instance.addPostFrameCallback((_) => _load(width));
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
