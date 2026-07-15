enum AppSection { races, history, reports }

class AppPageChrome {
  const AppPageChrome({required this.title, required this.subtitle});

  final String title;
  final String subtitle;
}

class AppWeb {
  AppWeb._();

  static Uri withAppMarker(Uri uri) {
    final query = Map<String, String>.from(uri.queryParameters);
    query['app'] = '1';
    return uri.replace(queryParameters: query);
  }

  static Uri topLevelUri(String baseUrl, AppSection section) {
    final base = Uri.parse(baseUrl.replaceFirst(RegExp(r'/+$'), ''));
    final basePath = _normalizedPath(base.path);
    final path = switch (section) {
      AppSection.races => basePath,
      AppSection.history => '$basePath/history',
      AppSection.reports => '$basePath/reports',
    };
    return withAppMarker(base.replace(path: path, queryParameters: const {}));
  }

  static Uri? normalizeInAppUrl(String rawUrl, String baseUrl) {
    final raw = rawUrl.trim();
    if (raw.isEmpty) {
      return null;
    }

    final base = Uri.parse(baseUrl.replaceFirst(RegExp(r'/+$'), ''));
    Uri normalized;
    try {
      final parsed = Uri.parse(raw);
      if (parsed.hasScheme) {
        normalized = parsed;
      } else if (raw.startsWith('/')) {
        normalized = base.replace(
          path: parsed.path,
          query: parsed.hasQuery ? parsed.query : null,
          fragment: parsed.hasFragment ? parsed.fragment : null,
        );
      } else {
        normalized = Uri.parse('$base/$raw');
      }
    } on FormatException {
      return null;
    }

    if (!isInternalUri(normalized, base)) {
      return null;
    }
    return withAppMarker(normalized);
  }

  static bool isInternalUri(Uri uri, Uri baseUri) {
    final scheme = uri.scheme.toLowerCase();
    return (scheme == 'http' || scheme == 'https') &&
        uri.host.toLowerCase() == baseUri.host.toLowerCase();
  }

  static AppSection sectionFor(Uri uri, Uri baseUri) {
    final path = _normalizedPath(uri.path);
    final basePath = _normalizedPath(baseUri.path);
    if (path == '$basePath/history') {
      return AppSection.history;
    }
    if (path == '$basePath/reports' || path.startsWith('$basePath/reports/')) {
      return AppSection.reports;
    }
    return AppSection.races;
  }

  static AppPageChrome chromeFor(Uri uri, Uri baseUri) {
    final path = _normalizedPath(uri.path);
    final basePath = _normalizedPath(baseUri.path);
    final section = sectionFor(uri, baseUri);
    final subtitle = switch (section) {
      AppSection.races => '当日公開レースと詳細ページを表示します',
      AppSection.history => '期間別の比較と振り返りを確認できます',
      AppSection.reports => '保存済みの日報アーカイブを確認できます',
    };
    final title = switch (path) {
      final value when value == basePath => 'レース一覧',
      final value when value.startsWith('$basePath/race/') => 'レース詳細',
      final value when value == '$basePath/history' => '履歴分析',
      final value when value == '$basePath/reports' => '私の日報',
      final value when value.startsWith('$basePath/reports/') => '日報詳細',
      _ => 'いかいもAI競馬',
    };
    return AppPageChrome(title: title, subtitle: subtitle);
  }

  static String _normalizedPath(String path) {
    if (path.isEmpty || path == '/') {
      return path.isEmpty ? '/' : path;
    }
    return path.replaceFirst(RegExp(r'/+$'), '');
  }
}
