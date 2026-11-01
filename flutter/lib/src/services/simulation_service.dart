import 'dart:convert';
import 'dart:io';
import '../config/app_config.dart';
import '../models/simulation_context.dart';

class SimulationService {
  HttpClient? _client;
  void cancel() {
    _client?.close(force: true);
    _client = null;
  }

  Future<Map<String, dynamic>> _get(
    String path,
    Map<String, String> query,
  ) async {
    cancel();
    final client = HttpClient()
      ..connectionTimeout = const Duration(seconds: 12);
    _client = client;
    try {
      final uri = Uri.parse(
        '${AppConfig.baseWebUrl}/api/public/simulator/$path',
      ).replace(queryParameters: query);
      final request = await client
          .getUrl(uri)
          .timeout(const Duration(seconds: 15));
      final response = await request.close().timeout(
        const Duration(seconds: 15),
      );
      final body = await utf8.decoder
          .bind(response)
          .join()
          .timeout(const Duration(seconds: 15));
      final payload = jsonDecode(body) as Map<String, dynamic>;
      if (payload['ok'] != true) {
        throw SimulationException(
          payload['error'] as String? ?? 'network_error',
        );
      }
      return payload['data'] as Map<String, dynamic>;
    } on SimulationException {
      rethrow;
    } on Object {
      throw const SimulationException('network_error');
    } finally {
      client.close(force: true);
      if (identical(client, _client)) _client = null;
    }
  }

  Future<Map<String, dynamic>> list(String date) =>
      _get('races', {'date': date});
  Future<Map<String, dynamic>> race(SimulationContext context) async {
    final data = await _get('races/${context.raceId}', {
      'date': context.date,
      if (context.inputVersion.isNotEmpty)
        'input_version': context.inputVersion,
    });
    if (data['race_id'] != context.raceId ||
        data['race_date'] != context.date ||
        data['active_model_version'] != context.modelVersion ||
        context.inputVersion.isNotEmpty &&
            data['input_version'] != context.inputVersion) {
      throw const SimulationException('identity_mismatch');
    }
    return data;
  }
}

class SimulationException implements Exception {
  const SimulationException(this.code);
  final String code;
  String get message => switch (code) {
    'unsupported' => 'このレースはシミュレーションの対象外です。',
    'unavailable' => '出走予定の馬が足りないため、試走できません。',
    'not_found' => 'このレースのデータはまだありません。',
    'preparing' => 'データを準備しています。しばらくしてから、もう一度お試しください。',
    'version_not_found' => '保存したデータが見つかりません。最新のデータを選んでください。',
    'disabled' => 'シミュレーションは現在ご利用いただけません。',
    _ => 'データを取得できませんでした。通信状況を確認して、もう一度お試しください。',
  };
}
