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
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import com.ikaimo.keiba.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val baseWebUri: Uri by lazy { Uri.parse(BuildConfig.BASE_WEB_URL) }
    private val baseHost: String by lazy { baseWebUri.host.orEmpty() }

    private var suppressBottomNavEvents = false
    private var currentTopLevel = TopLevelTab.RACES
    private var currentMorePage: MorePage = MorePage.PRIVACY
    private var moreDialog: AlertDialog? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)

        configureWebView()
        configureSwipeRefresh()
        configureBottomNavigation()
        configureBackHandling()

        if (savedInstanceState == null) {
            loadTopLevelPage(TopLevelTab.RACES, resetToRoot = true)
        } else {
            binding.webView.restoreState(savedInstanceState)
            syncChromeFromUrl(binding.webView.url)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        binding.webView.saveState(outState)
    }

    override fun onDestroy() {
        moreDialog?.dismiss()
        moreDialog = null
        binding.webView.apply {
            stopLoading()
            loadUrl("about:blank")
            clearHistory()
            removeAllViews()
            destroy()
        }
        super.onDestroy()
    }

    private fun configureWebView() {
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(binding.webView, true)

        binding.webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            loadsImagesAutomatically = true
            mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
            cacheMode = WebSettings.LOAD_DEFAULT
            useWideViewPort = true
            loadWithOverviewMode = true
            builtInZoomControls = false
            displayZoomControls = false
            setSupportZoom(false)
            userAgentString = "${userAgentString} IkaimoKeibaAndroid/0.1"
        }

        binding.webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                binding.progressIndicator.progress = newProgress
                binding.progressIndicator.isIndeterminate = false
                binding.progressIndicator.visibility = if (newProgress in 1..99) View.VISIBLE else View.GONE
                binding.swipeRefresh.isRefreshing = newProgress in 1..99 && binding.swipeRefresh.isRefreshing
            }
        }

        binding.webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val target = request?.url ?: return false
                if (!request.isForMainFrame) return false
                return handleUri(target)
            }

            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                binding.progressIndicator.visibility = View.VISIBLE
                syncChromeFromUrl(url)
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                binding.progressIndicator.visibility = View.GONE
                binding.swipeRefresh.isRefreshing = false
                syncChromeFromUrl(url)
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?,
            ) {
                if (request?.isForMainFrame != true) return
                binding.progressIndicator.visibility = View.GONE
                binding.swipeRefresh.isRefreshing = false
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
                    showMoreMenu()
                    false
                }

                else -> false
            }
        }

        binding.bottomNav.setOnItemReselectedListener { item ->
            if (suppressBottomNavEvents) return@setOnItemReselectedListener
            when (item.itemId) {
                R.id.menu_more -> showMoreMenu()
                else -> binding.webView.scrollTo(0, 0)
            }
        }
    }

    private fun configureBackHandling() {
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    val dialog = moreDialog
                    if (dialog?.isShowing == true) {
                        dialog.dismiss()
                        return
                    }
                    if (binding.webView.canGoBack()) {
                        binding.webView.goBack()
                    } else {
                        finish()
                    }
                }
            },
        )
    }

    private fun showMoreMenu() {
        val showingDialog = moreDialog
        if (showingDialog?.isShowing == true) {
            return
        }

        val labels = MorePage.entries.map { getString(it.labelRes) }.toTypedArray()
        val checkedIndex = MorePage.entries.indexOf(currentMorePage).coerceAtLeast(0)

        val dialog =
            MaterialAlertDialogBuilder(this)
            .setTitle(R.string.more_dialog_title)
            .setSingleChoiceItems(labels, checkedIndex, null)
            .setNegativeButton(android.R.string.cancel) { dialog, _ ->
                syncBottomNavigation(currentTopLevel)
                dialog.dismiss()
            }
            .setPositiveButton(R.string.open_page) { dialog, _ ->
                val selectedIndex =
                    (dialog as androidx.appcompat.app.AlertDialog).listView.checkedItemPosition
                        .coerceIn(0, MorePage.entries.lastIndex)
                currentMorePage = MorePage.entries[selectedIndex]
                loadMorePage(currentMorePage)
            }
            .setOnCancelListener {
                syncBottomNavigation(currentTopLevel)
            }
            .create()

        dialog.setCanceledOnTouchOutside(true)
        dialog.setOnDismissListener {
            moreDialog = null
            syncBottomNavigation(currentTopLevel)
        }
        moreDialog = dialog
        dialog.show()
    }

    private fun loadTopLevelPage(tab: TopLevelTab, resetToRoot: Boolean) {
        currentTopLevel = tab
        syncBottomNavigation(tab)

        if (!resetToRoot) return

        val targetUrl = when (tab) {
            TopLevelTab.RACES -> withAppMarker(BuildConfig.BASE_WEB_URL)
            TopLevelTab.HISTORY -> withAppMarker("${BuildConfig.BASE_WEB_URL}/history")
            TopLevelTab.REPORTS -> withAppMarker("${BuildConfig.BASE_WEB_URL}/reports")
            TopLevelTab.MORE -> withAppMarker(currentMorePage.url)
        }
        binding.webView.loadUrl(targetUrl)
    }

    private fun loadMorePage(page: MorePage) {
        currentTopLevel = TopLevelTab.MORE
        syncBottomNavigation(TopLevelTab.MORE)
        binding.webView.loadUrl(withAppMarker(page.url))
    }

    private fun syncBottomNavigation(tab: TopLevelTab) {
        val menuId = when (tab) {
            TopLevelTab.RACES -> R.id.menu_races
            TopLevelTab.HISTORY -> R.id.menu_history
            TopLevelTab.REPORTS -> R.id.menu_reports
            TopLevelTab.MORE -> R.id.menu_more
        }
        suppressBottomNavEvents = true
        binding.bottomNav.selectedItemId = menuId
        suppressBottomNavEvents = false
    }

    private fun syncChromeFromUrl(url: String?) {
        val uri = url?.let(Uri::parse) ?: return
        val topLevel = inferTopLevel(uri)
        currentTopLevel = topLevel
        if (topLevel == TopLevelTab.MORE) {
            currentMorePage = inferMorePage(uri) ?: currentMorePage
        }
        syncBottomNavigation(topLevel)
        binding.toolbarTitle.text = resolveTitle(uri)
        binding.toolbarSubtitle.text = resolveSubtitle(topLevel)
    }

    private fun inferTopLevel(uri: Uri): TopLevelTab {
        val path = uri.path.orEmpty().trimEnd('/')
        return when {
            path == "/keiba/history" -> TopLevelTab.HISTORY
            path.startsWith("/keiba/reports") -> TopLevelTab.REPORTS
            path == "/keiba/privacy" || path == "/keiba/terms" || path == "/keiba/disclaimer" || path == "/keiba/contact" ->
                TopLevelTab.MORE

            else -> TopLevelTab.RACES
        }
    }

    private fun inferMorePage(uri: Uri): MorePage? {
        return when (uri.path.orEmpty().trimEnd('/')) {
            "/keiba/privacy" -> MorePage.PRIVACY
            "/keiba/terms" -> MorePage.TERMS
            "/keiba/disclaimer" -> MorePage.DISCLAIMER
            "/keiba/contact" -> MorePage.CONTACT
            else -> null
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
            path == "/keiba/privacy" -> getString(R.string.more_privacy)
            path == "/keiba/terms" -> getString(R.string.more_terms)
            path == "/keiba/disclaimer" -> getString(R.string.more_disclaimer)
            path == "/keiba/contact" -> getString(R.string.more_contact)
            else -> getString(R.string.app_name)
        }
    }

    private fun resolveSubtitle(tab: TopLevelTab): String {
        return when (tab) {
            TopLevelTab.RACES -> getString(R.string.subtitle_races)
            TopLevelTab.HISTORY -> getString(R.string.subtitle_history)
            TopLevelTab.REPORTS -> getString(R.string.subtitle_reports)
            TopLevelTab.MORE -> getString(R.string.subtitle_more)
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
        val intent = Intent(Intent.ACTION_VIEW, uri).apply {
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
        val uri = Uri.parse(url)
        if (uri.getQueryParameter("app") == "1") {
            return url
        }
        return uri.buildUpon().appendQueryParameter("app", "1").build().toString()
    }

    private enum class TopLevelTab {
        RACES,
        HISTORY,
        REPORTS,
        MORE,
    }

    private enum class MorePage(val labelRes: Int, val url: String) {
        PRIVACY(R.string.more_privacy, "${BuildConfig.BASE_WEB_URL}/privacy"),
        TERMS(R.string.more_terms, "${BuildConfig.BASE_WEB_URL}/terms"),
        DISCLAIMER(R.string.more_disclaimer, "${BuildConfig.BASE_WEB_URL}/disclaimer"),
        CONTACT(R.string.more_contact, "${BuildConfig.BASE_WEB_URL}/contact"),
    }
}
