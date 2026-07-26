import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/services/app_startup_service.dart';

void main() {
  test('初始化成功时按顺序完成必要启动阶段', () async {
    final events = <String>[];
    final service = AppStartupService(
      initializeFirebase: () async => events.add('firebase'),
      onFirebaseReady: () async => events.add('crash-reporting'),
      initializeNativeConfiguration: () async => events.add('native'),
      onStage: (stage) => events.add(stage.name),
    );

    final result = await service.initialize();

    expect(result.firebaseReady, isTrue);
    expect(result.nativeConfigurationReady, isTrue);
    expect(events, [
      'firebaseInitialization',
      'firebase',
      'crash-reporting',
      'nativeConfiguration',
      'native',
      'complete',
    ]);
  });

  test('Firebase超时时仍继续原生配置并完成启动', () async {
    final failures = <AppStartupStage>[];
    var nativeInitialized = false;
    final neverCompletes = Completer<void>();
    final service = AppStartupService(
      initializeFirebase: () => neverCompletes.future,
      initializeNativeConfiguration: () async {
        nativeInitialized = true;
      },
      firebaseTimeout: const Duration(milliseconds: 1),
      onFailure: (stage, _, _) => failures.add(stage),
    );

    final result = await service.initialize();

    expect(result.firebaseReady, isFalse);
    expect(result.nativeConfigurationReady, isTrue);
    expect(nativeInitialized, isTrue);
    expect(failures, [AppStartupStage.firebaseInitialization]);
  });

  test('原生配置失败时仍返回可启动结果', () async {
    final failures = <AppStartupStage>[];
    final service = AppStartupService(
      initializeFirebase: () async {},
      initializeNativeConfiguration: () async {
        throw StateError('native unavailable');
      },
      onFailure: (stage, _, _) => failures.add(stage),
    );

    final result = await service.initialize();

    expect(result.firebaseReady, isTrue);
    expect(result.nativeConfigurationReady, isFalse);
    expect(failures, [AppStartupStage.nativeConfiguration]);
  });
}
