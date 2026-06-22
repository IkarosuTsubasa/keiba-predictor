package com.ikaimo.keiba.app

import android.graphics.Color
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding

object EdgeToEdgeUi {
    fun apply(
        activity: ComponentActivity,
        root: View,
        topInsetView: View,
        bottomInsetView: View? = null,
    ) {
        activity.enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
        )

        val initialRootPaddingLeft = root.paddingLeft
        val initialRootPaddingRight = root.paddingRight
        val initialTopPadding = topInsetView.paddingTop
        val initialBottomPadding = bottomInsetView?.paddingBottom ?: 0

        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars =
                insets.getInsets(
                    WindowInsetsCompat.Type.systemBars() or
                        WindowInsetsCompat.Type.displayCutout(),
                )
            view.updatePadding(
                left = initialRootPaddingLeft + bars.left,
                right = initialRootPaddingRight + bars.right,
            )
            topInsetView.updatePadding(top = initialTopPadding + bars.top)
            bottomInsetView?.updatePadding(bottom = initialBottomPadding + bars.bottom)
            insets
        }

        ViewCompat.requestApplyInsets(root)
    }
}
