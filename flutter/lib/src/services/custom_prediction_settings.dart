import 'package:shared_preferences/shared_preferences.dart';

import '../models/custom_prediction.dart';

class CustomPredictionSettingsSnapshot {
  const CustomPredictionSettingsSnapshot({
    this.parameters = const CustomPredictionParameters(),
    this.isConfigured = false,
  });

  final CustomPredictionParameters parameters;
  final bool isConfigured;
}

abstract interface class CustomPredictionSettingsStore {
  Future<CustomPredictionSettingsSnapshot> load();

  Future<void> save(CustomPredictionParameters parameters);
}

class CustomPredictionSettingsRepository
    implements CustomPredictionSettingsStore {
  CustomPredictionSettingsRepository({SharedPreferencesAsync? preferences})
    : _preferences = preferences ?? SharedPreferencesAsync();

  static const _abilityKey = 'custom_prediction.ability_level';
  static const _recentKey = 'custom_prediction.recent_level';
  static const _conditionKey = 'custom_prediction.condition_level';
  static const _opponentKey = 'custom_prediction.opponent_level';
  static const _paceJockeyKey = 'custom_prediction.pace_jockey_level';
  static const _pedigreeKey = 'custom_prediction.pedigree_level';
  static const _stabilityKey = 'custom_prediction.stability_level';
  static const _configuredKey = 'custom_prediction.configured';

  final SharedPreferencesAsync _preferences;

  @override
  Future<CustomPredictionSettingsSnapshot> load() async {
    final configured = await _preferences.getBool(_configuredKey) ?? false;
    final values = await Future.wait<int?>([
      _preferences.getInt(_abilityKey),
      _preferences.getInt(_recentKey),
      _preferences.getInt(_conditionKey),
      _preferences.getInt(_opponentKey),
      _preferences.getInt(_paceJockeyKey),
      _preferences.getInt(_pedigreeKey),
      _preferences.getInt(_stabilityKey),
    ]);
    return CustomPredictionSettingsSnapshot(
      isConfigured: configured,
      parameters: const CustomPredictionParameters().copyWith(
        abilityLevel: values[0] ?? 0,
        recentLevel: values[1] ?? 0,
        conditionLevel: values[2] ?? 0,
        opponentLevel: values[3] ?? 0,
        paceJockeyLevel: values[4] ?? 0,
        pedigreeLevel: values[5] ?? 0,
        stabilityLevel: values[6] ?? 0,
      ),
    );
  }

  @override
  Future<void> save(CustomPredictionParameters parameters) async {
    await Future.wait<void>([
      _preferences.setInt(_abilityKey, parameters.abilityLevel),
      _preferences.setInt(_recentKey, parameters.recentLevel),
      _preferences.setInt(_conditionKey, parameters.conditionLevel),
      _preferences.setInt(_opponentKey, parameters.opponentLevel),
      _preferences.setInt(_paceJockeyKey, parameters.paceJockeyLevel),
      _preferences.setInt(_pedigreeKey, parameters.pedigreeLevel),
      _preferences.setInt(_stabilityKey, parameters.stabilityLevel),
      _preferences.setBool(_configuredKey, true),
    ]);
  }
}
