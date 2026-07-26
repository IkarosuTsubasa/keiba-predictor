package com.ikaimo.keiba.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import android.webkit.WebSettings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugins.googlemobileads.GoogleMobileAdsPlugin

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        GoogleMobileAdsPlugin.registerNativeAdFactory(
            flutterEngine,
            NATIVE_AD_FACTORY,
            MoreNativeAdFactory(layoutInflater),
        )
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            PLATFORM_CHANNEL,
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "getRuntimeConfig" -> result.success(runtimeConfig())
                "getNotificationSettings" -> result.success(notificationSettings())
                "setNotificationSettings" -> {
                    val values = call.arguments as? Map<*, *> ?: emptyMap<Any, Any>()
                    notificationPreferences().edit()
                        .putBoolean(KEY_ENABLED, values[KEY_ENABLED] as? Boolean ?: true)
                        .putBoolean(KEY_CENTRAL_ENABLED, values[KEY_CENTRAL_ENABLED] as? Boolean ?: true)
                        .putBoolean(KEY_LOCAL_ENABLED, values[KEY_LOCAL_ENABLED] as? Boolean ?: true)
                        .apply()
                    result.success(null)
                }
                "shouldRequestNotificationPermission" -> result.success(shouldRequestNotificationPermission())
                "markNotificationPermissionPrompted" -> {
                    getSharedPreferences(PERMISSION_PREFS, Context.MODE_PRIVATE)
                        .edit().putBoolean(KEY_PROMPTED, true).apply()
                    result.success(null)
                }
                "getNotificationAuthorizationStatus" ->
                    result.success(notificationAuthorizationStatus())
                "openNotificationSettings" -> {
                    val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                        .putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
                    runCatching { startActivity(intent) }
                        .onSuccess { result.success(true) }
                        .onFailure { result.success(false) }
                }
                else -> result.notImplemented()
            }
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        GoogleMobileAdsPlugin.unregisterNativeAdFactory(flutterEngine, NATIVE_AD_FACTORY)
        super.cleanUpFlutterEngine(flutterEngine)
    }

    private fun runtimeConfig(): Map<String, String> =
        mapOf(
            "baseWebUrl" to BuildConfig.BASE_WEB_URL,
            "bannerAdUnitId" to BuildConfig.BANNER_AD_UNIT_ID,
            "nativeMoreAdUnitId" to BuildConfig.NATIVE_MORE_AD_UNIT_ID,
            "customPredictionRewardedAdUnitId" to BuildConfig.CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID,
            "fcmTopic" to BuildConfig.FCM_TOPIC,
            "fcmCentralTopic" to BuildConfig.FCM_CENTRAL_TOPIC,
            "fcmLocalTopic" to BuildConfig.FCM_LOCAL_TOPIC,
            "userAgent" to "${WebSettings.getDefaultUserAgent(this)} IkaimoKeibaAndroid/0.1",
            "appVersionName" to BuildConfig.VERSION_NAME,
            "appVersionCode" to BuildConfig.VERSION_CODE.toString(),
        )

    private fun notificationSettings(): Map<String, Boolean> =
        notificationPreferences().let { preferences ->
            mapOf(
                KEY_ENABLED to preferences.getBoolean(KEY_ENABLED, true),
                KEY_CENTRAL_ENABLED to preferences.getBoolean(KEY_CENTRAL_ENABLED, true),
                KEY_LOCAL_ENABLED to preferences.getBoolean(KEY_LOCAL_ENABLED, true),
            )
        }

    private fun notificationPreferences() =
        applicationContext.getSharedPreferences(NOTIFICATION_PREFS, Context.MODE_PRIVATE)

    private fun shouldRequestNotificationPermission(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return false
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return false
        return !getSharedPreferences(PERMISSION_PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_PROMPTED, false)
    }

    private fun notificationAuthorizationStatus(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return "authorized"
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            return "authorized"
        }
        val prompted = getSharedPreferences(PERMISSION_PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_PROMPTED, false)
        return if (prompted) "denied" else "notDetermined"
    }

    private companion object {
        const val PLATFORM_CHANNEL = "com.ikaimo.keiba.app/platform"
        const val NATIVE_AD_FACTORY = "moreNativeAd"
        const val NOTIFICATION_PREFS = "notification_preferences"
        const val PERMISSION_PREFS = "notification_permission"
        const val KEY_ENABLED = "enabled"
        const val KEY_CENTRAL_ENABLED = "central_enabled"
        const val KEY_LOCAL_ENABLED = "local_enabled"
        const val KEY_PROMPTED = "prompted_once"
    }
}
