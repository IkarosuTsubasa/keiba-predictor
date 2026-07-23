import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/services/reward_ad_requirement.dart';

void main() {
  group('RemoteConfigRewardAdRequirement', () {
    test('uses the exact iOS parameter with a false local default', () async {
      final client = _FakeRemoteConfigClient(value: true);
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => true,
      );

      expect(requirement.isRequired, isFalse);

      await requirement.initialize();

      expect(client.setConfigSettingsCalls, 1);
      expect(client.fetchTimeout, const Duration(seconds: 10));
      expect(client.minimumFetchInterval, const Duration(hours: 1));
      expect(client.defaults, const {'iOSRewardAD': false});
      expect(client.activateCalls, 1);
      expect(client.lastReadKey, 'iOSRewardAD');
      expect(requirement.isRequired, isTrue);
    });

    test('refresh fetches and applies the latest iOS value', () async {
      final client = _FakeRemoteConfigClient(value: false);
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => true,
      );
      await requirement.initialize();
      client.value = true;

      await requirement.refresh();

      expect(client.fetchAndActivateCalls, 1);
      expect(requirement.isRequired, isTrue);
    });

    test('concurrent refresh calls share one Remote Config fetch', () async {
      final fetchCompleter = Completer<bool>();
      final client = _FakeRemoteConfigClient(
        value: false,
        fetchCompleter: fetchCompleter,
      );
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => true,
      );
      await requirement.initialize();

      final first = requirement.refresh();
      final second = requirement.refresh();
      await Future<void>.delayed(Duration.zero);

      expect(identical(first, second), isTrue);
      expect(client.fetchAndActivateCalls, 1);

      client.value = true;
      fetchCompleter.complete(true);
      await Future.wait([first, second]);

      expect(requirement.isRequired, isTrue);
    });

    test(
      'a fetch with no newly activated value still reads active config',
      () async {
        final client = _FakeRemoteConfigClient(
          value: true,
          fetchAndActivateResult: false,
        );
        final requirement = RemoteConfigRewardAdRequirement(
          client: client,
          isIOS: () => true,
        );

        await requirement.refresh();

        expect(requirement.isRequired, isTrue);
      },
    );

    test('iOS safely falls back to false when defaults fail', () async {
      final client = _FakeRemoteConfigClient(
        value: true,
        setDefaultsError: StateError('defaults unavailable'),
      );
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => true,
      );

      await requirement.initialize();

      expect(requirement.isRequired, isFalse);
      expect(client.activateCalls, 0);
    });

    test(
      'iOS safely falls back to false when cached activation fails',
      () async {
        final client = _FakeRemoteConfigClient(
          value: true,
          activateError: StateError('activate unavailable'),
        );
        final requirement = RemoteConfigRewardAdRequirement(
          client: client,
          isIOS: () => true,
        );

        await requirement.initialize();

        expect(requirement.isRequired, isFalse);
      },
    );

    test('iOS safely falls back to false when refresh fails', () async {
      final client = _FakeRemoteConfigClient(value: true);
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => true,
      );
      await requirement.initialize();
      expect(requirement.isRequired, isTrue);
      client.fetchAndActivateError = StateError('fetch unavailable');

      await requirement.refresh();

      expect(requirement.isRequired, isFalse);
    });

    test('non-iOS always preserves the rewarded-ad requirement', () async {
      final client = _FakeRemoteConfigClient(value: false);
      final requirement = RemoteConfigRewardAdRequirement(
        client: client,
        isIOS: () => false,
      );

      expect(requirement.isRequired, isTrue);
      await requirement.initialize();
      await requirement.refresh();

      expect(requirement.isRequired, isTrue);
      expect(client.setConfigSettingsCalls, 0);
      expect(client.setDefaultsCalls, 0);
      expect(client.activateCalls, 0);
      expect(client.fetchAndActivateCalls, 0);
    });
  });
}

class _FakeRemoteConfigClient implements RewardAdRemoteConfigClient {
  _FakeRemoteConfigClient({
    required this.value,
    this.fetchAndActivateResult = true,
    this.fetchCompleter,
    this.setDefaultsError,
    this.activateError,
  });

  bool value;
  final bool fetchAndActivateResult;
  final Completer<bool>? fetchCompleter;
  final Object? setDefaultsError;
  final Object? activateError;
  Object? fetchAndActivateError;

  Map<String, Object>? defaults;
  String? lastReadKey;
  int setDefaultsCalls = 0;
  int setConfigSettingsCalls = 0;
  int activateCalls = 0;
  int fetchAndActivateCalls = 0;
  Duration? fetchTimeout;
  Duration? minimumFetchInterval;

  @override
  Future<void> setConfigSettings({
    required Duration fetchTimeout,
    required Duration minimumFetchInterval,
  }) async {
    setConfigSettingsCalls += 1;
    this.fetchTimeout = fetchTimeout;
    this.minimumFetchInterval = minimumFetchInterval;
  }

  @override
  Future<void> setDefaults(Map<String, Object> values) async {
    setDefaultsCalls += 1;
    final error = setDefaultsError;
    if (error != null) throw error;
    defaults = Map<String, Object>.from(values);
  }

  @override
  Future<bool> activate() async {
    activateCalls += 1;
    final error = activateError;
    if (error != null) throw error;
    return true;
  }

  @override
  Future<bool> fetchAndActivate() async {
    fetchAndActivateCalls += 1;
    final error = fetchAndActivateError;
    if (error != null) throw error;
    final completer = fetchCompleter;
    if (completer != null) return completer.future;
    return fetchAndActivateResult;
  }

  @override
  bool getBool(String key) {
    lastReadKey = key;
    return value;
  }
}
