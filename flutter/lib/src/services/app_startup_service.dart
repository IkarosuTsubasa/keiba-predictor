import 'dart:async';

enum AppStartupStage { firebaseInitialization, nativeConfiguration, complete }

typedef StartupAction = Future<void> Function();
typedef StartupStageCallback = void Function(AppStartupStage stage);
typedef StartupFailureCallback =
    void Function(AppStartupStage stage, Object error, StackTrace stack);

class AppStartupResult {
  const AppStartupResult({
    required this.firebaseReady,
    required this.nativeConfigurationReady,
  });

  final bool firebaseReady;
  final bool nativeConfigurationReady;
}

/// Runs only the services that must be ready before the main application UI.
///
/// Every stage is bounded and fails open so an unavailable SDK or native
/// channel cannot leave the app on the launch screen until iOS terminates it.
class AppStartupService {
  const AppStartupService({
    required this.initializeFirebase,
    required this.initializeNativeConfiguration,
    this.onFirebaseReady,
    this.onStage,
    this.onFailure,
    this.firebaseTimeout = const Duration(seconds: 5),
    this.nativeConfigurationTimeout = const Duration(seconds: 3),
  });

  final StartupAction initializeFirebase;
  final StartupAction initializeNativeConfiguration;
  final StartupAction? onFirebaseReady;
  final StartupStageCallback? onStage;
  final StartupFailureCallback? onFailure;
  final Duration firebaseTimeout;
  final Duration nativeConfigurationTimeout;

  Future<AppStartupResult> initialize() async {
    final firebaseReady = await _runStage(
      AppStartupStage.firebaseInitialization,
      initializeFirebase,
      firebaseTimeout,
    );
    if (firebaseReady) {
      await _runFirebaseReadyHook();
    }

    final nativeConfigurationReady = await _runStage(
      AppStartupStage.nativeConfiguration,
      initializeNativeConfiguration,
      nativeConfigurationTimeout,
    );
    onStage?.call(AppStartupStage.complete);

    return AppStartupResult(
      firebaseReady: firebaseReady,
      nativeConfigurationReady: nativeConfigurationReady,
    );
  }

  Future<bool> _runStage(
    AppStartupStage stage,
    StartupAction action,
    Duration timeout,
  ) async {
    onStage?.call(stage);
    try {
      await action().timeout(timeout);
      return true;
    } catch (error, stack) {
      onFailure?.call(stage, error, stack);
      return false;
    }
  }

  Future<void> _runFirebaseReadyHook() async {
    final callback = onFirebaseReady;
    if (callback == null) return;
    try {
      await callback().timeout(firebaseTimeout);
    } catch (error, stack) {
      onFailure?.call(AppStartupStage.firebaseInitialization, error, stack);
    }
  }
}
