package com.ikaimo.keiba.app

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.DownloadListener
import android.webkit.URLUtil
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.ads.AdListener
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.AdSize
import com.google.android.gms.ads.AdView
import com.google.android.gms.ads.LoadAdError
import com.google.android.gms.ads.MobileAds
import com.google.android.material.snackbar.Snackbar
import com.ikaimo.keiba.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val baseHost: String by lazy { Uri.parse(BuildConfig.BASE_WEB_URL).host.orEmpty() }

    private var suppressBottomNavEvents = false
    private var currentTopLevel = TopLevelTab.RACES
    private var launchOverlayDismissed = false
    private var bannerAdView: AdView? = null
    private var pendingOverrideUrl: String? = null
    private var pendingOverrideTitle: String? = null
    private val launchReloadRunnable =
        Runnable {
            if (launchOverlayDismissed) return@Runnable
            binding.launchOverlayHint.visibility = View.VISIBLE
            binding.launchOverlayReloadButton.visibility = View.VISIBLE
        }
    private val requestNotificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            NotificationPermissionHelper.markPrompted(this)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        EdgeToEdgeUi.apply(window, binding.root, binding.toolbar, binding.bottomChrome)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        binding.launchOverlayReloadButton.setOnClickListener {
            retryInitialLoad()
        }

        configureWebView()
        configureSwipeRefresh()
        configureBottomNavigation()
        configureBackHandling()
        MobileAds.initialize(this)
        configureBannerAd()
        maybeRequestNotificationPermission()

        launchOverlayDismissed =
            savedInstanceState?.getBoolean(STATE_LAUNCH_OVERLAY_DISMISSED, false) == true
        if (launchOverlayDismissed) {
            binding.launchOverlay.visibility = View.GONE
            binding.launchOverlay.alpha = 0f
        }

        if (savedInstanceState == null) {
            if (!handleStartDestination(intent, resetToRoot = true)) {
                loadTopLevelPage(TopLevelTab.RACES, resetToRoot = true)
            }
        } else {
            binding.webView.restoreState(savedInstanceState)
            syncChromeFromUrl(binding.webView.url)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleStartDestination(intent, resetToRoot = true)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(STATE_LAUNCH_OVERLAY_DISMISSED, launchOverlayDismissed)
        binding.webView.saveState(outState)
    }

    override fun onDestroy() {
        cancelLaunchReloadAffordance()
        bannerAdView?.destroy()
        bannerAdView = null
        binding.webView.apply {
            stopLoading()
            loadUrl("about:blank")
            clearHistory()
            removeAllViews()
            destroy()
        }
        super.onDestroy()
    }

    override fun onPause() {
        bannerAdView?.pause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        bannerAdView?.resume()
    }

    private fun configureWebView() {
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(binding.webView, true)

        binding.webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            loadsImagesAutomatically = true
            mixedContentMode =
                if (BuildConfig.ALLOW_INSECURE_WEB_CONTENT) {
                    WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
                } else {
                    WebSettings.MIXED_CONTENT_NEVER_ALLOW
                }
            cacheMode = WebSettings.LOAD_DEFAULT
            useWideViewPort = true
            loadWithOverviewMode = true
            builtInZoomControls = false
            displayZoomControls = false
            setSupportZoom(false)
            userAgentString = "${userAgentString} IkaimoKeibaAndroid/0.1"
        }

        binding.webView.webChromeClient =
            object : WebChromeClient() {
                override fun onProgressChanged(view: WebView?, newProgress: Int) {
                    binding.progressIndicator.progress = newProgress
                    binding.progressIndicator.isIndeterminate = false
                    binding.progressIndicator.visibility = if (newProgress in 1..99) View.VISIBLE else View.GONE
                    binding.swipeRefresh.isRefreshing = newProgress in 1..99 && binding.swipeRefresh.isRefreshing
                }
            }

        binding.webView.webViewClient =
            object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    val target = request?.url ?: return false
                    if (!request.isForMainFrame) return false
                    return handleUri(target)
                }

                override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                    binding.progressIndicator.visibility = View.VISIBLE
                    scheduleLaunchReloadAffordance()
                    syncChromeFromUrl(url)
                }

                override fun onPageFinished(view: WebView?, url: String?) {
                    binding.progressIndicator.visibility = View.GONE
                    binding.swipeRefresh.isRefreshing = false
                    syncChromeFromUrl(url)
                    dismissLaunchOverlay()
                }

                override fun onReceivedError(
                    view: WebView?,
                    request: WebResourceRequest?,
                    error: WebResourceError?,
                ) {
                    if (request?.isForMainFrame != true) return
                    binding.progressIndicator.visibility = View.GONE
                    binding.swipeRefresh.isRefreshing = false
                    dismissLaunchOverlay()
                    Snackbar.make(
                        binding.root,
                        getString(R.string.web_load_error),
                        Snackbar.LENGTH_LONG,
                    ).setAction(R.string.retry) {
                        binding.webView.reload()
                    }.show()
                }
            }

        binding.webView.setDownloadListener(
            DownloadListener { url, _, _, _, _ ->
                openExternal(Uri.parse(url))
            },
        )
    }

    private fun dismissLaunchOverlay() {
        if (launchOverlayDismissed) return
        launchOverlayDismissed = true
        cancelLaunchReloadAffordance()
        binding.launchOverlay.animate()
            .alpha(0f)
            .setDuration(220L)
            .withEndAction {
                binding.launchOverlay.visibility = View.GONE
            }
            .start()
    }

    private fun scheduleLaunchReloadAffordance() {
        if (launchOverlayDismissed) return
        cancelLaunchReloadAffordance()
        binding.launchOverlayHint.visibility = View.GONE
        binding.launchOverlayReloadButton.visibility = View.GONE
        binding.launchOverlay.postDelayed(launchReloadRunnable, LAUNCH_RELOAD_DELAY_MS)
    }

    private fun cancelLaunchReloadAffordance() {
        binding.launchOverlay.removeCallbacks(launchReloadRunnable)
    }

    private fun retryInitialLoad() {
        cancelLaunchReloadAffordance()
        binding.launchOverlayHint.visibility = View.GONE
        binding.launchOverlayReloadButton.visibility = View.GONE
        val currentUrl = binding.webView.url?.takeIf { it.isNotBlank() }
        val overrideUrl = pendingOverrideUrl?.takeIf { it.isNotBlank() }
        when {
            !currentUrl.isNullOrBlank() -> binding.webView.loadUrl(currentUrl)
            !overrideUrl.isNullOrBlank() -> binding.webView.loadUrl(overrideUrl)
            else -> loadTopLevelPage(currentTopLevel, resetToRoot = true)
        }
    }

    private fun configureBannerAd() {
        val adUnitId = BuildConfig.BANNER_AD_UNIT_ID.trim()
        if (adUnitId.isEmpty()) {
            binding.bannerHost.visibility = View.GONE
            return
        }

        val adView =
            AdView(this).apply {
                this.adUnitId = adUnitId
                adListener =
                    object : AdListener() {
                        override fun onAdLoaded() {
                            binding.bannerHost.visibility = View.VISIBLE
                        }

                        override fun onAdFailedToLoad(error: LoadAdError) {
                            binding.bannerHost.visibility = View.GONE
                        }
                    }
            }

        bannerAdView = adView
        binding.bannerHost.removeAllViews()
        binding.bannerHost.addView(
            adView,
            android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT,
            ),
        )

        binding.bannerHost.post {
            val adWidthPx =
                binding.bannerHost.width.takeIf { it > 0 } ?: resources.displayMetrics.widthPixels
            val density = resources.displayMetrics.density
            val adWidthDp = (adWidthPx / density).toInt().coerceAtLeast(1)
            adView.setAdSize(
                AdSize.getCurrentOrientationAnchoredAdaptiveBannerAdSize(
                    this,
                    adWidthDp,
                ),
            )
            adView.loadAd(AdRequest.Builder().build())
        }
    }

    private fun configureSwipeRefresh() {
        binding.swipeRefresh.setOnRefreshListener {
            binding.webView.reload()
        }
    }

    private fun configureBottomNavigation() {
        binding.bottomNav.setOnItemSelectedListener { item ->
            if (suppressBottomNavEvents) return@setOnItemSelectedListener true

            when (item.itemId) {
                R.id.menu_races -> {
                    loadTopLevelPage(TopLevelTab.RACES, resetToRoot = true)
                    true
                }

                R.id.menu_history -> {
                    loadTopLevelPage(TopLevelTab.HISTORY, resetToRoot = true)
                    true
                }

                R.id.menu_reports -> {
                    loadTopLevelPage(TopLevelTab.REPORTS, resetToRoot = true)
                    true
                }

                R.id.menu_more -> {
                    openMoreScreen()
                    false
                }

                else -> false
            }
        }

        binding.bottomNav.setOnItemReselectedListener { item ->
            if (suppressBottomNavEvents) return@setOnItemReselectedListener
            when (item.itemId) {
                R.id.menu_more -> openMoreScreen()
                else -> binding.webView.scrollTo(0, 0)
            }
        }
    }

    private fun openMoreScreen() {
        startActivity(Intent(this, MoreActivity::class.java))
    }

    private fun handleStartDestination(intent: Intent?, resetToRoot: Boolean): Boolean {
        if (handleExplicitWebDestination(intent)) {
            return true
        }

        val destination = intent?.getStringExtra(AppNavigation.EXTRA_START_DESTINATION) ?: return false
        val tab =
            when (destination) {
                AppNavigation.DEST_HISTORY -> TopLevelTab.HISTORY
                AppNavigation.DEST_REPORTS -> TopLevelTab.REPORTS
                else -> TopLevelTab.RACES
            }
        loadTopLevelPage(tab, resetToRoot = resetToRoot)
        intent.removeExtra(AppNavigation.EXTRA_START_DESTINATION)
        setIntent(intent)
        return true
    }

    private fun handleExplicitWebDestination(intent: Intent?): Boolean {
        intent?.let {
            val url = intent.getStringExtra(AppNavigation.EXTRA_WEB_URL)?.trim().orEmpty()
            if (url.isBlank()) return false

            val normalizedUrl = AppWeb.normalizeInAppUrl(url, BuildConfig.BASE_WEB_URL) ?: return false
            pendingOverrideUrl = normalizedUrl
            pendingOverrideTitle =
                intent.getStringExtra(AppNavigation.EXTRA_WEB_TITLE)?.trim()?.ifBlank { null }
            binding.webView.loadUrl(normalizedUrl)
            intent.removeExtra(AppNavigation.EXTRA_WEB_URL)
            intent.removeExtra(AppNavigation.EXTRA_WEB_TITLE)
            setIntent(intent)
        }
        return true
    }

    private fun maybeRequestNotificationPermission() {
        if (!NotificationPermissionHelper.shouldRequestOnLaunch(this)) return
        requestNotificationPermission.launch(android.Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun configureBackHandling() {
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (binding.webView.canGoBack()) {
                        binding.webView.goBack()
                    } else {
                        finish()
                    }
                }
            },
        )
    }

    private fun loadTopLevelPage(tab: TopLevelTab, resetToRoot: Boolean) {
        currentTopLevel = tab
        syncBottomNavigation(tab)

        if (!resetToRoot) return

        val targetUrl =
            when (tab) {
                TopLevelTab.RACES -> withAppMarker(BuildConfig.BASE_WEB_URL)
                TopLevelTab.HISTORY -> withAppMarker("${BuildConfig.BASE_WEB_URL}/history")
                TopLevelTab.REPORTS -> withAppMarker("${BuildConfig.BASE_WEB_URL}/reports")
            }
        binding.webView.loadUrl(targetUrl)
    }

    private fun syncBottomNavigation(tab: TopLevelTab) {
        val menuId =
            when (tab) {
                TopLevelTab.RACES -> R.id.menu_races
                TopLevelTab.HISTORY -> R.id.menu_history
                TopLevelTab.REPORTS -> R.id.menu_reports
            }
        suppressBottomNavEvents = true
        binding.bottomNav.selectedItemId = menuId
        suppressBottomNavEvents = false
    }

    private fun syncChromeFromUrl(url: String?) {
        val uri = url?.let(Uri::parse) ?: return
        val topLevel = inferTopLevel(uri)
        currentTopLevel = topLevel
        syncBottomNavigation(topLevel)
        val overrideTitle =
            if (pendingOverrideUrl == url) {
                pendingOverrideTitle
            } else {
                pendingOverrideUrl = null
                pendingOverrideTitle = null
                null
            }
        binding.toolbarTitle.text = overrideTitle ?: resolveTitle(uri)
        binding.toolbarSubtitle.text = resolveSubtitle(topLevel)
    }

    private fun inferTopLevel(uri: Uri): TopLevelTab {
        val path = uri.path.orEmpty().trimEnd('/')
        return when {
            path == "/keiba/history" -> TopLevelTab.HISTORY
            path.startsWith("/keiba/reports") -> TopLevelTab.REPORTS
            else -> TopLevelTab.RACES
        }
    }

    private fun resolveTitle(uri: Uri): String {
        val path = uri.path.orEmpty().trimEnd('/')
        return when {
            path == "/keiba" -> getString(R.string.title_races)
            path.startsWith("/keiba/race/") -> getString(R.string.title_race_detail)
            path == "/keiba/history" -> getString(R.string.title_history)
            path == "/keiba/reports" -> getString(R.string.title_reports)
            path.startsWith("/keiba/reports/") -> getString(R.string.title_report_detail)
            else -> getString(R.string.app_name)
        }
    }

    private fun resolveSubtitle(tab: TopLevelTab): String {
        return when (tab) {
            TopLevelTab.RACES -> getString(R.string.subtitle_races)
            TopLevelTab.HISTORY -> getString(R.string.subtitle_history)
            TopLevelTab.REPORTS -> getString(R.string.subtitle_reports)
        }
    }

    private fun handleUri(uri: Uri): Boolean {
        val scheme = uri.scheme.orEmpty().lowercase()

        if (scheme == "mailto" || scheme == "tel") {
            openExternal(uri)
            return true
        }

        if (scheme == "http" || scheme == "https") {
            if (uri.host == baseHost) {
                return false
            }
            openExternal(uri)
            return true
        }

        openExternal(uri)
        return true
    }

    private fun openExternal(uri: Uri) {
        val intent =
            Intent(Intent.ACTION_VIEW, uri).apply {
                addCategory(Intent.CATEGORY_BROWSABLE)
            }
        try {
            startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            Snackbar.make(
                binding.root,
                getString(R.string.external_open_error, URLUtil.guessFileName(uri.toString(), null, null)),
                Snackbar.LENGTH_LONG,
            ).show()
        }
    }

    private fun withAppMarker(url: String): String {
        return AppWeb.withAppMarker(url)
    }

    private enum class TopLevelTab {
        RACES,
        HISTORY,
        REPORTS,
    }

    private companion object {
        const val STATE_LAUNCH_OVERLAY_DISMISSED = "state_launch_overlay_dismissed"
        const val LAUNCH_RELOAD_DELAY_MS = 10_000L
    }
}
