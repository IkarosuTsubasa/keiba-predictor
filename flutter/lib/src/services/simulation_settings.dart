import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/simulation_context.dart';

class SimulationSettings {
  static const _prefix = 'simulation.v1.';
  Future<Map<String, dynamic>?> load(String key) async {
    try {
      final value = (await SharedPreferences.getInstance()).getString(
        '$_prefix$key',
      );
      return value == null ? null : jsonDecode(value) as Map<String, dynamic>;
    } on Object {
      return null;
    }
  }

  Future<void> save(String key, Map<String, dynamic> value) async {
    await (await SharedPreferences.getInstance()).setString(
      '$_prefix$key',
      jsonEncode(value),
    );
  }

  Future<void> saveScene(
    SimulationContext context,
    Map<String, dynamic> state,
  ) async {
    await save('scene.${context.storageKey}', state);
    await save('recent', context.toJson());
    final settings = state['settings'] as Map<String, dynamic>? ?? {};
    await save('preferences', {
      'speed': settings['speed'] ?? 4,
      'view': settings['view'] ?? 'course',
    });
  }
}
