package com.ikaimo.keiba.app

import android.content.Context
import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging

enum class NotificationScope {
    CENTRAL,
    LOCAL,
    GENERAL,
}

object NotificationPreferences {
    private const val PREFS_NAME = "notification_preferences"
    private const val KEY_ENABLED = "enabled"
    private const val KEY_CENTRAL_ENABLED = "central_enabled"
    private const val KEY_LOCAL_ENABLED = "local_enabled"
    private const val TAG = "NotificationPrefs"

    fun isEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_ENABLED, true)

    fun isCentralEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_CENTRAL_ENABLED, true)

    fun isLocalEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_LOCAL_ENABLED, true)

    fun setEnabled(context: Context, enabled: Boolean) {
        prefs(context).edit().putBoolean(KEY_ENABLED, enabled).apply()
    }

    fun setCentralEnabled(context: Context, enabled: Boolean) {
        prefs(context).edit().putBoolean(KEY_CENTRAL_ENABLED, enabled).apply()
    }

    fun setLocalEnabled(context: Context, enabled: Boolean) {
        prefs(context).edit().putBoolean(KEY_LOCAL_ENABLED, enabled).apply()
    }

    fun shouldShow(context: Context, scope: NotificationScope): Boolean {
        if (!isEnabled(context)) return false
        return when (scope) {
            NotificationScope.CENTRAL -> isCentralEnabled(context)
            NotificationScope.LOCAL -> isLocalEnabled(context)
            NotificationScope.GENERAL -> true
        }
    }

    fun scopeFromData(data: Map<String, String>): NotificationScope {
        val candidates =
            listOf(
                data["notification_scope"],
                data["scope_group"],
                data["scope_key"],
                data["race_scope"],
                data["source"],
            )

        for (candidate in candidates) {
            val value = candidate?.trim()?.lowercase().orEmpty()
            when {
                value in setOf("local", "nar") || value.contains("地方") -> return NotificationScope.LOCAL
                value.startsWith("central") || value == "jra" || value.contains("中央") -> return NotificationScope.CENTRAL
            }
        }
        return NotificationScope.GENERAL
    }

    fun syncFcmTopics(context: Context) {
        val enabled = isEnabled(context)
        val centralEnabled = enabled && isCentralEnabled(context)
        val localEnabled = enabled && isLocalEnabled(context)

        syncTopic(BuildConfig.FCM_TOPIC, enabled, "general")
        syncTopic(BuildConfig.FCM_CENTRAL_TOPIC, centralEnabled, "central")
        syncTopic(BuildConfig.FCM_LOCAL_TOPIC, localEnabled, "local")
    }

    private fun syncTopic(topic: String, shouldSubscribe: Boolean, label: String) {
        val normalizedTopic = topic.trim()
        if (normalizedTopic.isBlank()) return

        val task =
            if (shouldSubscribe) {
                FirebaseMessaging.getInstance().subscribeToTopic(normalizedTopic)
            } else {
                FirebaseMessaging.getInstance().unsubscribeFromTopic(normalizedTopic)
            }

        task.addOnCompleteListener { result ->
            if (BuildConfig.DEBUG && !result.isSuccessful) {
                Log.w(TAG, "FCM $label topic sync failed: $normalizedTopic", result.exception)
            }
        }
    }

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
}
