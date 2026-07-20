import 'dart:async';

import 'package:google_mobile_ads/google_mobile_ads.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../config/app_config.dart';
import 'ad_consent_service.dart';
import 'native_platform_bridge.dart';

enum CustomPredictionRewardResult { rewarded, dismissed, unavailable, failed }

abstract interface class CustomPredictionAccessStore {
  Future<DateTime?> loadUnlockedUntil();

  Future<void> saveUnlockedUntil(DateTime value);
}

class CustomPredictionAccessRepository implements CustomPredictionAccessStore {
  CustomPredictionAccessRepository({SharedPreferencesAsync? preferences})
    : _preferences = preferences ?? SharedPreferencesAsync();

  static const _unlockedUntilKey =
      'custom_prediction.rewarded_access_until_epoch_ms';

  final SharedPreferencesAsync _preferences;

  @override
  Future<DateTime?> loadUnlockedUntil() async {
    final milliseconds = await _preferences.getInt(_unlockedUntilKey);
    if (milliseconds == null || milliseconds <= 0) return null;
    return DateTime.fromMillisecondsSinceEpoch(milliseconds, isUtc: true);
  }

  @override
  Future<void> saveUnlockedUntil(DateTime value) {
    return _preferences.setInt(
      _unlockedUntilKey,
      value.toUtc().millisecondsSinceEpoch,
    );
  }
}

abstract interface class CustomPredictionRewardedGateway {
  Future<CustomPredictionRewardResult> showRewardedAd();
}

typedef _RewardedAdLoader = Future<RewardedAd?> Function(String adUnitId);

class GoogleCustomPredictionRewardedGateway
    implements CustomPredictionRewardedGateway {
  GoogleCustomPredictionRewardedGateway({
    AdConsentService? consentService,
    bool Function()? adsEnabled,
    String Function()? adUnitId,
    Future<RewardedAd?> Function(String adUnitId)? loadAd,
  }) : _consentService = consentService ?? AdConsentService.instance,
       _adsEnabled = adsEnabled ?? (() => AppConfig.adsEnabled),
       _adUnitId =
           adUnitId ??
           (() {
             return NativePlatformBridge
                     .runtimeConfig
                     ?.customPredictionRewardedAdUnitId ??
                 '';
           }),
       _loadAd = loadAd ?? _defaultLoadAd;

  final AdConsentService _consentService;
  final bool Function() _adsEnabled;
  final String Function() _adUnitId;
  final _RewardedAdLoader _loadAd;

  @override
  Future<CustomPredictionRewardResult> showRewardedAd() async {
    if (!_adsEnabled()) return CustomPredictionRewardResult.unavailable;

    try {
      await _consentService.initialize();
    } catch (_) {
      return CustomPredictionRewardResult.unavailable;
    }
    if (!_consentService.canRequestAds.value) {
      return CustomPredictionRewardResult.unavailable;
    }

    final unitId = _adUnitId().trim();
    if (unitId.isEmpty) return CustomPredictionRewardResult.unavailable;

    RewardedAd? ad;
    try {
      ad = await _loadAd(unitId).timeout(const Duration(seconds: 20));
    } catch (_) {
      return CustomPredictionRewardResult.failed;
    }
    final loadedAd = ad;
    if (loadedAd == null) return CustomPredictionRewardResult.failed;

    final result = Completer<CustomPredictionRewardResult>();
    var rewardEarned = false;

    void complete(CustomPredictionRewardResult value) {
      if (!result.isCompleted) result.complete(value);
    }

    loadedAd.fullScreenContentCallback = FullScreenContentCallback<RewardedAd>(
      onAdDismissedFullScreenContent: (dismissedAd) {
        dismissedAd.dispose();
        complete(
          rewardEarned
              ? CustomPredictionRewardResult.rewarded
              : CustomPredictionRewardResult.dismissed,
        );
      },
      onAdFailedToShowFullScreenContent: (failedAd, _) {
        failedAd.dispose();
        complete(CustomPredictionRewardResult.failed);
      },
    );

    try {
      await loadedAd.show(
        onUserEarnedReward: (_, _) {
          rewardEarned = true;
        },
      );
    } catch (_) {
      loadedAd.dispose();
      complete(CustomPredictionRewardResult.failed);
    }

    return result.future.timeout(
      const Duration(minutes: 5),
      onTimeout: () {
        loadedAd.dispose();
        return CustomPredictionRewardResult.failed;
      },
    );
  }

  static Future<RewardedAd?> _defaultLoadAd(String adUnitId) {
    final result = Completer<RewardedAd?>();
    RewardedAd.load(
      adUnitId: adUnitId,
      request: const AdRequest(),
      rewardedAdLoadCallback: RewardedAdLoadCallback(
        onAdLoaded: (ad) {
          if (result.isCompleted) {
            ad.dispose();
            return;
          }
          result.complete(ad);
        },
        onAdFailedToLoad: (_) {
          if (!result.isCompleted) result.complete(null);
        },
      ),
    );
    return result.future;
  }
}
