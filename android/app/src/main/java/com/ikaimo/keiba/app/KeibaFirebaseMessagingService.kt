package com.ikaimo.keiba.app

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class KeibaFirebaseMessagingService : FirebaseMessagingService() {

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        if (BuildConfig.DEBUG) {
            Log.i(TAG, "FCM token refreshed: $token")
        }
    }

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        NotificationChannels.ensureCreated(this)

        val title =
            message.notification?.title
                ?: message.data["title"]
                ?: getString(R.string.app_name)
        val body =
            message.notification?.body
                ?: message.data["body"]
                ?: return

        val rawUrl = message.data["url"]?.trim().orEmpty()
        val webUrl = rawUrl.takeIf { it.isNotBlank() }?.let { AppWeb.normalizeInAppUrl(it, BuildConfig.BASE_WEB_URL) }
        val destination = message.data["destination"]?.trim().orEmpty().ifBlank { null }
        val contentIntent = NotificationRouting.contentIntent(this, webUrl, title, destination)

        if (!NotificationPermissionHelper.isGranted(this)) {
            Log.w(TAG, "Notification permission not granted; skip foreground notification display")
            return
        }

        val notification =
            NotificationCompat.Builder(this, NotificationChannels.GENERAL_CHANNEL_ID)
                .setSmallIcon(android.R.drawable.sym_def_app_icon)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(NotificationCompat.BigTextStyle().bigText(body))
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                .setContentIntent(contentIntent)
                .build()

        if (
            ActivityCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "POST_NOTIFICATIONS is not granted; skip notification display")
            return
        }

        try {
            NotificationManagerCompat.from(this).notify(nextNotificationId(), notification)
        } catch (error: SecurityException) {
            Log.w(TAG, "Notification display failed after permission check", error)
        }
    }

    private fun nextNotificationId(): Int {
        return (System.currentTimeMillis() and 0xFFFFFFF).toInt()
    }

    private companion object {
        const val TAG = "KeibaFCM"
    }
}
