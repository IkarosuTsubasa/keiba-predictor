package com.ikaimo.keiba.app

import android.app.PendingIntent
import android.content.Context
import android.content.Intent

object NotificationRouting {
    fun contentIntent(
        context: Context,
        webUrl: String?,
        title: String?,
        destination: String?,
    ): PendingIntent {
        val intent =
            Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                when {
                    !webUrl.isNullOrBlank() -> {
                        putExtra(AppNavigation.EXTRA_WEB_URL, webUrl)
                        if (!title.isNullOrBlank()) {
                            putExtra(AppNavigation.EXTRA_WEB_TITLE, title)
                        }
                    }

                    !destination.isNullOrBlank() -> {
                        putExtra(AppNavigation.EXTRA_START_DESTINATION, destination)
                    }
                }
            }

        return PendingIntent.getActivity(
            context,
            intent.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }
}
