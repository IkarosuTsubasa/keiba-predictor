import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import '../config/app_config.dart';

typedef _ConsentFormPresenter = Future<FormError?> Function();
typedef _MobileAdsInitializer = Future<void> Function();

class AdConsentService {
  AdConsentService({
    ConsentInformation? consentInformation,
    Future<FormError?> Function()? loadRequiredConsentForm,
    Future<FormError?> Function()? showPrivacyOptionsForm,
    Future<void> Function()? initializeMobileAds,
    bool Function()? adsEnabled,
  }) : _consentInformation = consentInformation ?? ConsentInformation.instance,
       _loadRequiredConsentForm =
           loadRequiredConsentForm ?? _defaultLoadRequiredConsentForm,
       _showPrivacyOptionsForm =
           showPrivacyOptionsForm ?? _defaultShowPrivacyOptionsForm,
       _initializeMobileAds =
           initializeMobileAds ?? _defaultInitializeMobileAds,
       _adsEnabled = adsEnabled ?? (() => AppConfig.adsEnabled);

  static final instance = AdConsentService();

  final ConsentInformation _consentInformation;
  final _ConsentFormPresenter _loadRequiredConsentForm;
  final _ConsentFormPresenter _showPrivacyOptionsForm;
  final _MobileAdsInitializer _initializeMobileAds;
  final bool Function() _adsEnabled;

  final canRequestAds = ValueNotifier<bool>(false);
  final privacyOptionsRequired = ValueNotifier<bool>(false);

  Future<void>? _startupInitialization;
  Future<void>? _consentFlowInProgress;
  Future<void>? _mobileAdsInitialization;
  bool _needsNetworkRecovery = true;

  Future<void> initialize() {
    if (!_adsEnabled()) return Future.value();
    return _startupInitialization ??= _runConsentFlow();
  }

  /// Re-runs UMP only when its previous attempt could not establish whether
  /// ads may be requested. This is used after iOS returns from its first-use
  /// network permission alert.
  Future<void> retryAfterNetworkRecovery() {
    if (!_adsEnabled()) return Future.value();
    final existing = _consentFlowInProgress;
    if (existing != null) return existing;
    if (canRequestAds.value || !_needsNetworkRecovery) {
      return Future.value();
    }
    return _runConsentFlow();
  }

  Future<void> _runConsentFlow() {
    final existing = _consentFlowInProgress;
    if (existing != null) return existing;

    late final Future<void> operation;
    operation = _initializeConsentFlow().whenComplete(() {
      if (identical(_consentFlowInProgress, operation)) {
        _consentFlowInProgress = null;
      }
    });
    _consentFlowInProgress = operation;
    return operation;
  }

  Future<void> _initializeConsentFlow() async {
    final consentInfoUpdated = await _requestConsentInfoUpdate();
    FormError? formError;
    if (consentInfoUpdated) {
      formError = await _guardedFormPresentation(_loadRequiredConsentForm);
      if (formError != null) {
        debugPrint(
          'Ad consent form could not be presented: ${formError.message}',
        );
      }
    }
    final eligibilityChecked = await _refreshEligibility();
    _needsNetworkRecovery =
        !eligibilityChecked ||
        (!canRequestAds.value && (!consentInfoUpdated || formError != null));
  }

  Future<bool> _requestConsentInfoUpdate() async {
    final completer = Completer<bool>();
    try {
      _consentInformation.requestConsentInfoUpdate(
        ConsentRequestParameters(),
        () {
          if (!completer.isCompleted) completer.complete(true);
        },
        (error) {
          debugPrint(
            'Ad consent information could not be updated: ${error.message}',
          );
          if (!completer.isCompleted) completer.complete(false);
        },
      );
      return await completer.future;
    } catch (error) {
      debugPrint('Ad consent information is unavailable: $error');
      return false;
    }
  }

  Future<FormError?> _guardedFormPresentation(
    _ConsentFormPresenter presenter,
  ) async {
    try {
      return await presenter();
    } catch (error) {
      debugPrint('Ad privacy form is unavailable: $error');
      return FormError(errorCode: 0, message: error.toString());
    }
  }

  Future<bool> _refreshEligibility() async {
    try {
      final requirement = await _consentInformation
          .getPrivacyOptionsRequirementStatus();
      privacyOptionsRequired.value =
          requirement == PrivacyOptionsRequirementStatus.required;
    } catch (error) {
      debugPrint('Ad privacy options status is unavailable: $error');
      privacyOptionsRequired.value = false;
    }

    try {
      if (!await _consentInformation.canRequestAds()) {
        canRequestAds.value = false;
        return true;
      }
      await _ensureMobileAdsInitialized();
      canRequestAds.value = true;
      return true;
    } catch (error) {
      debugPrint('Ads remain disabled because consent is unavailable: $error');
      canRequestAds.value = false;
      return false;
    }
  }

  Future<void> _ensureMobileAdsInitialized() async {
    final existing = _mobileAdsInitialization;
    if (existing != null) {
      await existing;
      return;
    }

    final initialization = _initializeMobileAds();
    _mobileAdsInitialization = initialization;
    try {
      await initialization;
    } catch (_) {
      _mobileAdsInitialization = null;
      rethrow;
    }
  }

  Future<bool> showPrivacyOptions() async {
    if (!_adsEnabled()) return false;
    final formError = await _guardedFormPresentation(_showPrivacyOptionsForm);
    await _refreshEligibility();
    if (formError != null) {
      debugPrint(
        'Ad privacy options could not be presented: ${formError.message}',
      );
    }
    return formError == null;
  }

  @visibleForTesting
  void dispose() {
    canRequestAds.dispose();
    privacyOptionsRequired.dispose();
  }

  static Future<FormError?> _defaultLoadRequiredConsentForm() async {
    FormError? formError;
    await ConsentForm.loadAndShowConsentFormIfRequired((error) {
      formError = error;
    });
    return formError;
  }

  static Future<FormError?> _defaultShowPrivacyOptionsForm() async {
    FormError? formError;
    await ConsentForm.showPrivacyOptionsForm((error) {
      formError = error;
    });
    return formError;
  }

  static Future<void> _defaultInitializeMobileAds() async {
    await MobileAds.instance.initialize();
  }
}
