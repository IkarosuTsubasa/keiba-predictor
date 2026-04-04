package com.ikaimo.keiba.app

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.URLUtil
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.snackbar.Snackbar
import com.ikaimo.keiba.app.databinding.ActivityStaticPageBinding

class StaticPageActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStaticPageBinding
    private val baseHost: String by lazy { Uri.parse(BuildConfig.BASE_WEB_URL).host.orEmpty() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStaticPageBinding.inflate(layoutInflater)
        setContentView(binding.root)
        EdgeToEdgeUi.apply(window, binding.root, binding.toolbar)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        binding.toolbarTitle.text = intent.getStringExtra(AppNavigation.EXTRA_PAGE_TITLE).orEmpty()
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }

        configureWebView()
        configureSwipeRefresh()
        configureBackHandling()

        if (savedInstanceState == null) {
            val targetUrl = intent.getStringExtra(AppNavigation.EXTRA_PAGE_URL).orEmpty()
            if (targetUrl.isBlank()) {
                finish()
                return
            }
            binding.webView.loadUrl(targetUrl)
        } else {
            binding.webView.restoreState(savedInstanceState)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        binding.webView.saveState(outState)
    }

    override fun onDestroy() {
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
                }

                override fun onPageFinished(view: WebView?, url: String?) {
                    binding.progressIndicator.visibility = View.GONE
                    binding.swipeRefresh.isRefreshing = false
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
    }

    private fun configureSwipeRefresh() {
        binding.swipeRefresh.setOnRefreshListener {
            binding.webView.reload()
        }
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

    companion object {
        fun newIntent(context: Context, pageTitle: String, pageUrl: String): Intent =
            Intent(context, StaticPageActivity::class.java).apply {
                putExtra(AppNavigation.EXTRA_PAGE_TITLE, pageTitle)
                putExtra(AppNavigation.EXTRA_PAGE_URL, pageUrl)
            }
    }
}
