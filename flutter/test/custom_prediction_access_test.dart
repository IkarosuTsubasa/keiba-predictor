import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/services/ad_consent_service.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_access.dart';

void main() {
  test('リワード広告は読み込み前に消音し、終了後に元へ戻す', () async {
    final consentService = AdConsentService(adsEnabled: () => false);
    consentService.canRequestAds.value = true;
    final muteChanges = <bool>[];

    final gateway = GoogleCustomPredictionRewardedGateway(
      consentService: consentService,
      adsEnabled: () => true,
      adUnitId: () => 'test-rewarded-unit',
      loadAd: (_) async => null,
      setAppMuted: (muted) async => muteChanges.add(muted),
    );

    final result = await gateway.showRewardedAd();

    expect(result, CustomPredictionRewardResult.failed);
    expect(muteChanges, [true, false]);
    consentService.dispose();
  });
}
