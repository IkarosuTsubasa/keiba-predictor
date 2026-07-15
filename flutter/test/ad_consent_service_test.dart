import 'package:flutter_test/flutter_test.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';
import 'package:ikaimo_keiba/src/services/ad_consent_service.dart';

void main() {
  test('同意情報と必須フォームの確認後にだけ広告を有効化する', () async {
    final consentInformation = _FakeConsentInformation(
      canRequestAdsValue: true,
      privacyOptionsRequirementStatus: PrivacyOptionsRequirementStatus.required,
    );
    var formPresentationCount = 0;
    var mobileAdsInitializationCount = 0;
    final service = AdConsentService(
      consentInformation: consentInformation,
      loadRequiredConsentForm: () async {
        formPresentationCount += 1;
        return null;
      },
      initializeMobileAds: () async {
        mobileAdsInitializationCount += 1;
      },
      adsEnabled: () => true,
    );

    await service.initialize();
    await service.initialize();

    expect(consentInformation.updateCount, 1);
    expect(formPresentationCount, 1);
    expect(mobileAdsInitializationCount, 1);
    expect(service.canRequestAds.value, isTrue);
    expect(service.privacyOptionsRequired.value, isTrue);
    service.dispose();
  });

  test('同意情報の更新に失敗しても有効なキャッシュがあれば広告を許可する', () async {
    final consentInformation = _FakeConsentInformation(
      updateSucceeds: false,
      canRequestAdsValue: true,
    );
    var formPresentationCount = 0;
    var mobileAdsInitializationCount = 0;
    final service = AdConsentService(
      consentInformation: consentInformation,
      loadRequiredConsentForm: () async {
        formPresentationCount += 1;
        return null;
      },
      initializeMobileAds: () async {
        mobileAdsInitializationCount += 1;
      },
      adsEnabled: () => true,
    );

    await service.initialize();

    expect(formPresentationCount, 0);
    expect(mobileAdsInitializationCount, 1);
    expect(service.canRequestAds.value, isTrue);
    service.dispose();
  });

  test('UMP が広告を許可しない間は Mobile Ads を初期化しない', () async {
    final consentInformation = _FakeConsentInformation(
      canRequestAdsValue: false,
    );
    var mobileAdsInitializationCount = 0;
    final service = AdConsentService(
      consentInformation: consentInformation,
      loadRequiredConsentForm: () async => null,
      initializeMobileAds: () async {
        mobileAdsInitializationCount += 1;
      },
      adsEnabled: () => true,
    );

    await service.initialize();

    expect(mobileAdsInitializationCount, 0);
    expect(service.canRequestAds.value, isFalse);
    service.dispose();
  });
}

class _FakeConsentInformation implements ConsentInformation {
  _FakeConsentInformation({
    this.updateSucceeds = true,
    required this.canRequestAdsValue,
    this.privacyOptionsRequirementStatus =
        PrivacyOptionsRequirementStatus.notRequired,
  });

  final bool updateSucceeds;
  final bool canRequestAdsValue;
  final PrivacyOptionsRequirementStatus privacyOptionsRequirementStatus;
  int updateCount = 0;

  @override
  Future<bool> canRequestAds() async => canRequestAdsValue;

  @override
  Future<ConsentStatus> getConsentStatus() async =>
      canRequestAdsValue ? ConsentStatus.obtained : ConsentStatus.required;

  @override
  Future<PrivacyOptionsRequirementStatus>
  getPrivacyOptionsRequirementStatus() async => privacyOptionsRequirementStatus;

  @override
  Future<bool> isConsentFormAvailable() async => true;

  @override
  void requestConsentInfoUpdate(
    ConsentRequestParameters params,
    OnConsentInfoUpdateSuccessListener successListener,
    OnConsentInfoUpdateFailureListener failureListener,
  ) {
    updateCount += 1;
    if (updateSucceeds) {
      successListener();
    } else {
      failureListener(FormError(errorCode: 1, message: 'network unavailable'));
    }
  }

  @override
  Future<void> reset() async {}
}
