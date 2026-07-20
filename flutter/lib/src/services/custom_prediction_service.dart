import 'dart:convert';
import 'dart:io';

import '../config/app_config.dart';
import '../models/custom_prediction.dart';

abstract interface class CustomPredictionDataSource {
  Future<CustomPredictionDay> fetch();
}

class CustomPredictionService implements CustomPredictionDataSource {
  const CustomPredictionService();

  @override
  Future<CustomPredictionDay> fetch() async {
    final endpoint = Uri.parse(
      '${AppConfig.baseWebUrl}/api/public/custom-predictions',
    );
    final client = HttpClient()
      ..connectionTimeout = const Duration(seconds: 12);
    try {
      final request = await client
          .getUrl(endpoint)
          .timeout(const Duration(seconds: 15));
      request.headers.set(HttpHeaders.acceptHeader, 'application/json');
      final response = await request.close().timeout(
        const Duration(seconds: 15),
      );
      final body = await utf8.decoder.bind(response).join();
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw const CustomPredictionException('予測データを取得できませんでした。');
      }
      final decoded = jsonDecode(body);
      if (decoded is! Map) {
        throw const CustomPredictionException('予測データの形式が正しくありません。');
      }
      final payload = decoded.map(
        (key, value) => MapEntry(key.toString(), value),
      );
      if (payload['ok'] != true) {
        throw const CustomPredictionException('予測データを取得できませんでした。');
      }
      return CustomPredictionDay.fromApi(payload);
    } on CustomPredictionException {
      rethrow;
    } on FormatException {
      throw const CustomPredictionException('予測データの形式が正しくありません。');
    } on Object {
      throw const CustomPredictionException('通信状況を確認して、もう一度お試しください。');
    } finally {
      client.close(force: true);
    }
  }
}

class CustomPredictionException implements Exception {
  const CustomPredictionException(this.message);

  final String message;

  @override
  String toString() => message;
}
