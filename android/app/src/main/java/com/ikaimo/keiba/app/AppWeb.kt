package com.ikaimo.keiba.app

import androidx.core.net.toUri

object AppWeb {
    fun normalizeInAppUrl(url: String, baseUrl: String): String? {
        val baseUri = baseUrl.toUri()
        val rawUri = url.toUri()

        val normalized =
            when {
                rawUri.scheme.isNullOrBlank() && url.startsWith("/") ->
                    "${baseUri.scheme}://${baseUri.host}$url".toUri()

                rawUri.scheme.isNullOrBlank() ->
                    "$baseUrl/$url".toUri()

                else -> rawUri
            }

        if (normalized.host != baseUri.host) {
            return null
        }

        return withAppMarker(normalized.toString())
    }

    fun withAppMarker(url: String): String {
        val uri = url.toUri()
        if (uri.getQueryParameter("app") == "1") {
            return url
        }
        return uri.buildUpon().appendQueryParameter("app", "1").build().toString()
    }
}
