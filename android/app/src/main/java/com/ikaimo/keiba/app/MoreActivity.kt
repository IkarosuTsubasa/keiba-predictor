package com.ikaimo.keiba.app

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.google.android.gms.ads.AdListener
import com.google.android.gms.ads.AdLoader
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.LoadAdError
import com.google.android.gms.ads.MobileAds
import com.google.android.gms.ads.nativead.MediaView
import com.google.android.gms.ads.nativead.NativeAd
import com.google.android.gms.ads.nativead.NativeAdOptions
import com.google.android.gms.ads.nativead.NativeAdView
import com.ikaimo.keiba.app.databinding.ActivityMoreBinding
import com.ikaimo.keiba.app.databinding.ItemMorePageBinding

class MoreActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMoreBinding
    private var nativeAd: NativeAd? = null
    private var suppressBottomNavEvents = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMoreBinding.inflate(layoutInflater)
        setContentView(binding.root)
        EdgeToEdgeUi.apply(window, binding.root, binding.toolbar, binding.bottomNav)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)

        configureOptions()
        configureBottomNavigation()
        MobileAds.initialize(this)
        configureNativeAd()
    }

    override fun onDestroy() {
        nativeAd?.destroy()
        nativeAd = null
        super.onDestroy()
    }

    private fun configureOptions() {
        binding.moreOptionsContainer.removeAllViews()
        MorePage.entries.forEach { page ->
            val itemBinding =
                ItemMorePageBinding.inflate(LayoutInflater.from(this), binding.moreOptionsContainer, true)
            itemBinding.moreItemTitle.setText(page.titleRes)
            itemBinding.moreItemDescription.setText(page.descriptionRes)
            itemBinding.root.setOnClickListener {
                startActivity(
                    StaticPageActivity.newIntent(
                        context = this,
                        pageTitle = getString(page.titleRes),
                        pageUrl = page.appUrl(),
                    ),
                )
            }
        }
    }

    private fun configureBottomNavigation() {
        suppressBottomNavEvents = true
        binding.bottomNav.selectedItemId = R.id.menu_more
        suppressBottomNavEvents = false

        binding.bottomNav.setOnItemSelectedListener { item ->
            if (suppressBottomNavEvents) return@setOnItemSelectedListener true

            when (item.itemId) {
                R.id.menu_races -> {
                    navigateToMain(AppNavigation.DEST_RACES)
                    true
                }

                R.id.menu_history -> {
                    navigateToMain(AppNavigation.DEST_HISTORY)
                    true
                }

                R.id.menu_reports -> {
                    navigateToMain(AppNavigation.DEST_REPORTS)
                    true
                }

                R.id.menu_more -> true
                else -> false
            }
        }

        binding.bottomNav.setOnItemReselectedListener { }
    }

    private fun navigateToMain(destination: String) {
        val intent =
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                putExtra(AppNavigation.EXTRA_START_DESTINATION, destination)
            }
        startActivity(intent)
        finish()
    }

    private fun configureNativeAd() {
        val adUnitId = BuildConfig.NATIVE_MORE_AD_UNIT_ID.trim()
        if (adUnitId.isEmpty()) {
            binding.nativeAdHost.isVisible = false
            return
        }

        val adLoader =
            AdLoader.Builder(this, adUnitId)
                .forNativeAd { ad ->
                    nativeAd?.destroy()
                    nativeAd = ad
                    val adView =
                        layoutInflater.inflate(
                            R.layout.view_native_more_ad,
                            binding.nativeAdHost,
                            false,
                        ) as NativeAdView
                    populateNativeAdView(ad, adView)
                    binding.nativeAdHost.removeAllViews()
                    binding.nativeAdHost.addView(adView)
                    binding.nativeAdHost.isVisible = true
                }
                .withNativeAdOptions(NativeAdOptions.Builder().build())
                .withAdListener(
                    object : AdListener() {
                        override fun onAdFailedToLoad(error: LoadAdError) {
                            binding.nativeAdHost.removeAllViews()
                            binding.nativeAdHost.isVisible = false
                        }
                    },
                )
                .build()

        adLoader.loadAd(AdRequest.Builder().build())
    }

    private fun populateNativeAdView(ad: NativeAd, adView: NativeAdView) {
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

        val mediaContent = ad.mediaContent
        mediaView.mediaContent = mediaContent
        mediaView.isVisible =
            mediaContent != null &&
                (
                    mediaContent.hasVideoContent() ||
                        mediaContent.mainImage != null
                )

        headlineView.text = ad.headline

        val bodyText = ad.body
        bodyView.text = bodyText
        bodyView.isVisible = !bodyText.isNullOrBlank()

        val ctaText = ad.callToAction
        callToActionView.text = ctaText
        callToActionView.isVisible = !ctaText.isNullOrBlank()

        val iconDrawable = ad.icon?.drawable
        if (iconDrawable != null) {
            iconView.setImageDrawable(iconDrawable)
            iconView.isVisible = true
        } else {
            iconView.setImageDrawable(null)
            iconView.isVisible = false
        }

        adView.setNativeAd(ad)
    }
}
