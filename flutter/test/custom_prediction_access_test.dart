import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/services/ad_consent_service.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_access.dart';

void main() {
  test('iOSのリワード広告は読み込み前に消音し、終了後に解除する', () async {
    final consentService = AdConsentService(adsEnabled: () => false);
    consentService.canRequestAds.value = true;
    final muteChanges = <bool>[];

    final gateway = GoogleCustomPredictionRewardedGateway(
      consentService: consentService,
      adsEnabled: () => true,
      adUnitId: () => 'test-rewarded-unit',
      loadAd: (_) async => null,
      setAppMuted: (muted) async => muteChanges.add(muted),
      muteRewardedAd: () => true,
    );

    final result = await gateway.showRewardedAd();

    expect(result, CustomPredictionRewardResult.failed);
    expect(muteChanges, [true, false]);
    consentService.dispose();
  });

  test('Androidのリワード広告は既存どおりアプリ全体を消音しない', () async {
    final consentService = AdConsentService(adsEnabled: () => false);
    consentService.canRequestAds.value = true;
    final muteChanges = <bool>[];

    final gateway = GoogleCustomPredictionRewardedGateway(
      consentService: consentService,
      adsEnabled: () => true,
      adUnitId: () => 'test-rewarded-unit',
      loadAd: (_) async => null,
      setAppMuted: (muted) async => muteChanges.add(muted),
      muteRewardedAd: () => false,
    );

    final result = await gateway.showRewardedAd();

    expect(result, CustomPredictionRewardResult.failed);
    expect(muteChanges, isEmpty);
    consentService.dispose();
  });
}
