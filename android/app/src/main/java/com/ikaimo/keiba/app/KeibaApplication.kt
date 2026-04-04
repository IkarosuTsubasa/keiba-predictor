package com.ikaimo.keiba.app

import android.app.Application
import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging

class KeibaApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        NotificationChannels.ensureCreated(this)
        FirebaseMessaging.getInstance().subscribeToTopic(BuildConfig.FCM_TOPIC).addOnCompleteListener { task ->
            if (BuildConfig.DEBUG && !task.isSuccessful) {
                Log.w(TAG, "FCM topic subscribe failed", task.exception)
            }
        }
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                if (BuildConfig.DEBUG) {
                    Log.w(TAG, "FCM token fetch failed", task.exception)
                }
                return@addOnCompleteListener
            }
            val token = task.result.orEmpty()
            if (BuildConfig.DEBUG && token.isNotBlank()) {
                Log.i(TAG, "FCM token: $token")
            }
        }
    }

    private companion object {
        const val TAG = "KeibaApplication"
    }
}
