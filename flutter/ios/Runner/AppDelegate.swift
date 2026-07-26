import Flutter
import UIKit
import UserNotifications
import WebKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private enum ConfigKey {
    static let bannerAdUnitID = "KEIBA_BANNER_AD_UNIT_ID"
    static let nativeMoreAdUnitID = "KEIBA_NATIVE_MORE_AD_UNIT_ID"
    static let customPredictionRewardedAdUnitID =
      "KEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID"
    static let fcmTopic = "KEIBA_FCM_TOPIC"
    static let fcmCentralTopic = "KEIBA_FCM_CENTRAL_TOPIC"
    static let fcmLocalTopic = "KEIBA_FCM_LOCAL_TOPIC"
  }

  private enum PreferenceKey {
    static let enabled = "enabled"
    static let centralEnabled = "central_enabled"
    static let localEnabled = "local_enabled"
    static let permissionPrompted = "notification_permission_prompted_once"
  }

  private var platformChannel: FlutterMethodChannel?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    UNUserNotificationCenter.current().delegate = self
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    let channel = FlutterMethodChannel(
      name: "com.ikaimo.keiba.app/platform",
      binaryMessenger: engineBridge.applicationRegistrar.messenger()
    )
    channel.setMethodCallHandler { [weak self] call, result in
      guard let self else {
        result(FlutterError(code: "app_delegate_unavailable", message: nil, details: nil))
        return
      }
      self.handlePlatformCall(call, result: result)
    }
    platformChannel = channel
  }

  private func handlePlatformCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "getRuntimeConfig":
      runtimeConfig(result: result)
    case "getNotificationSettings":
      result(notificationSettings())
    case "setNotificationSettings":
      let values = call.arguments as? [String: Any] ?? [:]
      let defaults = notificationDefaults
      defaults.set(values[PreferenceKey.enabled] as? Bool ?? true, forKey: PreferenceKey.enabled)
      defaults.set(
        values[PreferenceKey.centralEnabled] as? Bool ?? true,
        forKey: PreferenceKey.centralEnabled
      )
      defaults.set(
        values[PreferenceKey.localEnabled] as? Bool ?? true,
        forKey: PreferenceKey.localEnabled
      )
      result(nil)
    case "shouldRequestNotificationPermission":
      shouldRequestNotificationPermission(result: result)
    case "markNotificationPermissionPrompted":
      notificationDefaults.set(true, forKey: PreferenceKey.permissionPrompted)
      result(nil)
    case "getNotificationAuthorizationStatus":
      notificationAuthorizationStatus(result: result)
    case "openNotificationSettings":
      guard let url = URL(string: UIApplication.openSettingsURLString) else {
        result(false)
        return
      }
      UIApplication.shared.open(url, options: [:]) { opened in
        DispatchQueue.main.async {
          result(opened)
        }
      }
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func runtimeConfig(result: @escaping FlutterResult) {
    #if DEBUG
      let baseWebURL = "http://localhost:8000/keiba"
    #else
      let baseWebURL = "https://www.ikaimo-ai.com/keiba"
    #endif

    let webView = WKWebView(frame: .zero)
    webView.evaluateJavaScript("navigator.userAgent") { value, _ in
      let systemUserAgent = value as? String ?? ""
      let userAgent = [systemUserAgent, "IkaimoKeibaIOS/0.1"]
        .filter { !$0.isEmpty }
        .joined(separator: " ")
      result([
        "baseWebUrl": baseWebURL,
        "bannerAdUnitId": self.infoValue(ConfigKey.bannerAdUnitID),
        "nativeMoreAdUnitId": self.infoValue(ConfigKey.nativeMoreAdUnitID),
        "customPredictionRewardedAdUnitId": self.infoValue(
          ConfigKey.customPredictionRewardedAdUnitID
        ),
        "fcmTopic": self.infoValue(ConfigKey.fcmTopic),
        "fcmCentralTopic": self.infoValue(ConfigKey.fcmCentralTopic),
        "fcmLocalTopic": self.infoValue(ConfigKey.fcmLocalTopic),
        "userAgent": userAgent,
        "appVersionName": self.infoValue("CFBundleShortVersionString"),
        "appVersionCode": self.infoValue("CFBundleVersion"),
      ])
    }
  }

  private func infoValue(_ key: String) -> String {
    (Bundle.main.object(forInfoDictionaryKey: key) as? String)?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
  }

  private var notificationDefaults: UserDefaults {
    UserDefaults(suiteName: "com.ikaimo.keiba.app.notifications") ?? .standard
  }

  private func notificationSettings() -> [String: Bool] {
    let defaults = notificationDefaults
    return [
      PreferenceKey.enabled: boolValue(
        forKey: PreferenceKey.enabled,
        defaults: defaults,
        fallback: true
      ),
      PreferenceKey.centralEnabled: boolValue(
        forKey: PreferenceKey.centralEnabled,
        defaults: defaults,
        fallback: true
      ),
      PreferenceKey.localEnabled: boolValue(
        forKey: PreferenceKey.localEnabled,
        defaults: defaults,
        fallback: true
      ),
    ]
  }

  private func boolValue(forKey key: String, defaults: UserDefaults, fallback: Bool) -> Bool {
    defaults.object(forKey: key) == nil ? fallback : defaults.bool(forKey: key)
  }

  private func shouldRequestNotificationPermission(result: @escaping FlutterResult) {
    let prompted = notificationDefaults.bool(forKey: PreferenceKey.permissionPrompted)
    guard !prompted else {
      result(false)
      return
    }
    UNUserNotificationCenter.current().getNotificationSettings { settings in
      DispatchQueue.main.async {
        result(settings.authorizationStatus == .notDetermined)
      }
    }
  }

  private func notificationAuthorizationStatus(result: @escaping FlutterResult) {
    UNUserNotificationCenter.current().getNotificationSettings { settings in
      let value: String
      switch settings.authorizationStatus {
      case .authorized:
        value = "authorized"
      case .denied:
        value = "denied"
      case .notDetermined:
        value = "notDetermined"
      case .provisional:
        value = "provisional"
      case .ephemeral:
        value = "ephemeral"
      @unknown default:
        value = "unknown"
      }
      DispatchQueue.main.async {
        result(value)
      }
    }
  }
}
