package com.ikaimo.keiba.app

import androidx.annotation.StringRes

enum class MorePage(
    @StringRes val titleRes: Int,
    @StringRes val descriptionRes: Int,
    private val path: String,
) {
    PRIVACY(R.string.more_privacy, R.string.more_privacy_description, "privacy"),
    TERMS(R.string.more_terms, R.string.more_terms_description, "terms"),
    DISCLAIMER(R.string.more_disclaimer, R.string.more_disclaimer_description, "disclaimer"),
    CONTACT(R.string.more_contact, R.string.more_contact_description, "contact"),
    ;

    fun appUrl(): String = AppWeb.withAppMarker("${BuildConfig.BASE_WEB_URL}/$path")
}
