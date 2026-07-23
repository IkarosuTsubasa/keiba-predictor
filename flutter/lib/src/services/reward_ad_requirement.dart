import 'package:firebase_remote_config/firebase_remote_config.dart';
import 'package:flutter/foundation.dart';

/// Whether access to My予測 must be unlocked with a rewarded ad.
abstract interface class RewardAdRequirement {
  bool get isRequired;

  Future<void> refresh();
}

/// Minimal adapter around Remote Config so the policy can be tested without
/// constructing Firebase SDK objects.
abstract interface class RewardAdRemoteConfigClient {
  Future<void> setConfigSettings({
    required Duration fetchTimeout,
    required Duration minimumFetchInterval,
  });

  Future<void> setDefaults(Map<String, Object> defaults);

  Future<bool> activate();

  Future<bool> fetchAndActivate();

  bool getBool(String key);
}

class FirebaseRewardAdRemoteConfigClient implements RewardAdRemoteConfigClient {
  FirebaseRewardAdRemoteConfigClient([FirebaseRemoteConfig? remoteConfig])
    : _remoteConfig = remoteConfig ?? FirebaseRemoteConfig.instance;

  final FirebaseRemoteConfig _remoteConfig;

  @override
  Future<void> setConfigSettings({
    required Duration fetchTimeout,
    required Duration minimumFetchInterval,
  }) {
    return _remoteConfig.setConfigSettings(
      RemoteConfigSettings(
        fetchTimeout: fetchTimeout,
        minimumFetchInterval: minimumFetchInterval,
      ),
    );
  }

  @override
  Future<void> setDefaults(Map<String, Object> defaults) {
    return _remoteConfig.setDefaults(defaults);
  }

  @override
  Future<bool> activate() => _remoteConfig.activate();

  @override
  Future<bool> fetchAndActivate() => _remoteConfig.fetchAndActivate();

  @override
  bool getBool(String key) => _remoteConfig.getBool(key);
}

class RemoteConfigRewardAdRequirement implements RewardAdRequirement {
  factory RemoteConfigRewardAdRequirement({
    RewardAdRemoteConfigClient? client,
    bool Function()? isIOS,
  }) {
    return RemoteConfigRewardAdRequirement._(client, isIOS ?? _defaultIsIOS);
  }

  RemoteConfigRewardAdRequirement._(this._client, this._isIOS);

  static const parameterName = 'iOSRewardAD';
  static const fetchTimeout = Duration(seconds: 10);
  static const minimumFetchInterval = Duration(hours: 1);

  static final instance = RemoteConfigRewardAdRequirement();

  RewardAdRemoteConfigClient? _client;
  final bool Function() _isIOS;

  bool _clientReady = false;
  bool? _iosRemoteValue;
  Future<void>? _refreshInProgress;

  /// iOS fails open when Remote Config is unavailable. Other platforms keep
  /// the existing behavior and always require the rewarded ad.
  @override
  bool get isRequired => _isIOS() ? (_iosRemoteValue ?? false) : true;

  /// Makes the local default available, then activates any value fetched by a
  /// previous app session. No network request is performed here.
  Future<void> initialize() async {
    if (!_isIOS()) return;

    _iosRemoteValue = false;
    if (!await _ensureClientReady()) return;

    try {
      final client = _remoteConfigClient;
      await client.activate();
      _iosRemoteValue = client.getBool(parameterName);
    } catch (_) {
      _iosRemoteValue = false;
    }
  }

  /// Fetches and activates the latest value. Any fetch, activation, or value
  /// read failure safely falls back to not requiring the ad on iOS.
  @override
  Future<void> refresh() {
    if (!_isIOS()) return Future<void>.value();

    final existing = _refreshInProgress;
    if (existing != null) return existing;

    late final Future<void> operation;
    operation = _refreshOnce().whenComplete(() {
      if (identical(_refreshInProgress, operation)) {
        _refreshInProgress = null;
      }
    });
    _refreshInProgress = operation;
    return operation;
  }

  Future<void> _refreshOnce() async {
    if (!await _ensureClientReady()) {
      _iosRemoteValue = false;
      return;
    }

    try {
      final client = _remoteConfigClient;
      await client.fetchAndActivate();
      _iosRemoteValue = client.getBool(parameterName);
    } catch (_) {
      _iosRemoteValue = false;
    }
  }

  Future<bool> _ensureClientReady() async {
    if (_clientReady) return true;
    try {
      final client = _remoteConfigClient;
      await client.setConfigSettings(
        fetchTimeout: fetchTimeout,
        minimumFetchInterval: minimumFetchInterval,
      );
      await client.setDefaults(const {parameterName: false});
      _clientReady = true;
      return true;
    } catch (_) {
      _clientReady = false;
      return false;
    }
  }

  RewardAdRemoteConfigClient get _remoteConfigClient {
    return _client ??= FirebaseRewardAdRemoteConfigClient();
  }

  static bool _defaultIsIOS() {
    return !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;
  }
}
