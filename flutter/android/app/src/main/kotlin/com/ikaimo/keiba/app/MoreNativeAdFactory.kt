package com.ikaimo.keiba.app

import android.view.LayoutInflater
import android.view.View
import android.view.ContextThemeWrapper
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.google.android.gms.ads.nativead.MediaView
import com.google.android.gms.ads.nativead.NativeAd
import com.google.android.gms.ads.nativead.NativeAdView
import io.flutter.plugins.googlemobileads.NativeAdFactory

class MoreNativeAdFactory(
    private val layoutInflater: LayoutInflater,
) : NativeAdFactory {
    override fun createNativeAd(
        nativeAd: NativeAd,
        customOptions: MutableMap<String, Any>?,
    ): NativeAdView {
        val themedContext =
            ContextThemeWrapper(layoutInflater.context, R.style.NativeMoreAdTheme)
        val adView =
            LayoutInflater.from(themedContext)
                .inflate(R.layout.view_native_more_ad, null) as NativeAdView
        val mediaView = adView.findViewById<MediaView>(R.id.nativeAdMedia)
        val headlineView = adView.findViewById<TextView>(R.id.nativeAdHeadline)
        val bodyView = adView.findViewById<TextView>(R.id.nativeAdBody)
        val callToActionView = adView.findViewById<Button>(R.id.nativeAdCallToAction)
        val iconView = adView.findViewById<ImageView>(R.id.nativeAdIcon)

        adView.mediaView = mediaView
        adView.headlineView = headlineView
        adView.bodyView = bodyView
        adView.callToActionView = callToActionView
        adView.iconView = iconView

        val mediaContent = nativeAd.mediaContent
        mediaView.mediaContent = mediaContent
        mediaView.visibility =
            if (
                mediaContent != null &&
                (mediaContent.hasVideoContent() || mediaContent.mainImage != null)
            ) {
                View.VISIBLE
            } else {
                View.GONE
            }

        headlineView.text = nativeAd.headline
        bodyView.text = nativeAd.body
        bodyView.visibility = if (nativeAd.body.isNullOrBlank()) View.GONE else View.VISIBLE

        callToActionView.text = nativeAd.callToAction
        callToActionView.visibility =
            if (nativeAd.callToAction.isNullOrBlank()) View.GONE else View.VISIBLE

        val iconDrawable = nativeAd.icon?.drawable
        if (iconDrawable == null) {
            iconView.setImageDrawable(null)
            iconView.visibility = View.GONE
        } else {
            iconView.setImageDrawable(iconDrawable)
            iconView.visibility = View.VISIBLE
        }

        adView.setNativeAd(nativeAd)
        return adView
    }
}
