package com.ikaimo.keiba.app

import android.view.View
import android.view.Window
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding

object EdgeToEdgeUi {
    fun apply(
        window: Window,
        root: View,
        topInsetView: View,
        bottomInsetView: View? = null,
    ) {
        WindowCompat.setDecorFitsSystemWindows(window, false)

        val initialTopPadding = topInsetView.paddingTop
        val initialBottomPadding = bottomInsetView?.paddingBottom ?: 0

        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            topInsetView.updatePadding(top = initialTopPadding + systemBars.top)
            bottomInsetView?.updatePadding(bottom = initialBottomPadding + systemBars.bottom)
            insets
        }

        ViewCompat.requestApplyInsets(root)
    }
}
