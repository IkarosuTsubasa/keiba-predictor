import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.gms.google-services")
    id("com.google.firebase.crashlytics")
}

val localProperties =
    Properties().apply {
        val localPropertiesFile = rootProject.file("local.properties")
        if (localPropertiesFile.exists()) {
            localPropertiesFile.inputStream().use(::load)
        }
    }

fun localProperty(name: String): String? =
    localProperties.getProperty(name)?.trim()?.ifBlank { null }

val releaseBaseWebUrl = "https://www.ikaimo-ai.com/keiba"
val googleTestAdmobAppId = "ca-app-pub-3940256099942544~3347511713"
val googleTestBannerAdUnitId = "ca-app-pub-3940256099942544/9214589741"
val googleTestNativeAdUnitId = "ca-app-pub-3940256099942544/2247696110"
val admobAppId =
    providers.gradleProperty("KEIBA_ADMOB_APP_ID")
        .orElse(localProperty("KEIBA_ADMOB_APP_ID") ?: googleTestAdmobAppId)
        .get()
val debugBaseWebUrl =
    providers.gradleProperty("KEIBA_DEBUG_BASE_WEB_URL")
        .orElse(localProperty("KEIBA_DEBUG_BASE_WEB_URL") ?: "http://10.0.2.2:8000/keiba")
        .get()
val bannerAdUnitId =
    providers.gradleProperty("KEIBA_BANNER_AD_UNIT_ID")
        .orElse(localProperty("KEIBA_BANNER_AD_UNIT_ID") ?: googleTestBannerAdUnitId)
        .get()
val nativeMoreAdUnitId =
    providers.gradleProperty("KEIBA_NATIVE_MORE_AD_UNIT_ID")
        .orElse(localProperty("KEIBA_NATIVE_MORE_AD_UNIT_ID") ?: googleTestNativeAdUnitId)
        .get()
val fcmTopic =
    providers.gradleProperty("KEIBA_FCM_TOPIC")
        .orElse(localProperty("KEIBA_FCM_TOPIC") ?: "keiba-public-updates")
        .get()
val mobileApiToken =
    providers.gradleProperty("KEIBA_MOBILE_API_TOKEN")
        .orElse(localProperty("KEIBA_MOBILE_API_TOKEN") ?: "")
        .get()

val validateReleaseAdmobConfig =
    tasks.register("validateReleaseAdmobConfig") {
        group = "verification"
        description = "Fails the release build if AdMob is still using missing or test IDs."
        doLast {
            val problems = mutableListOf<String>()
            if (admobAppId.isBlank() || admobAppId == googleTestAdmobAppId) {
                problems += "KEIBA_ADMOB_APP_ID must be set to your production AdMob app ID for release builds."
            }
            if (bannerAdUnitId.isBlank() || bannerAdUnitId == googleTestBannerAdUnitId) {
                problems += "KEIBA_BANNER_AD_UNIT_ID must be set to your production banner ad unit ID for release builds."
            }
            if (nativeMoreAdUnitId.isBlank() || nativeMoreAdUnitId == googleTestNativeAdUnitId) {
                problems += "KEIBA_NATIVE_MORE_AD_UNIT_ID must be set to your production native ad unit ID for release builds."
            }
            if (problems.isNotEmpty()) {
                throw GradleException(problems.joinToString(separator = "\n"))
            }
        }
    }

tasks.configureEach {
    if (name.contains("Release", ignoreCase = true) && name != validateReleaseAdmobConfig.name) {
        dependsOn(validateReleaseAdmobConfig)
    }
}

android {
    namespace = "com.ikaimo.keiba.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ikaimo.keiba.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "BASE_WEB_URL", "\"$releaseBaseWebUrl\"")
        buildConfigField("String", "BANNER_AD_UNIT_ID", "\"$bannerAdUnitId\"")
        buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", "\"$nativeMoreAdUnitId\"")
        buildConfigField("String", "FCM_TOPIC", "\"$fcmTopic\"")
        buildConfigField("String", "MOBILE_API_TOKEN", "\"$mobileApiToken\"")
        buildConfigField("boolean", "ALLOW_INSECURE_WEB_CONTENT", "false")
        manifestPlaceholders["usesCleartextTraffic"] = "false"
        manifestPlaceholders["admobAppId"] = admobAppId
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            buildConfigField("String", "BASE_WEB_URL", "\"$debugBaseWebUrl\"")
            buildConfigField("String", "BANNER_AD_UNIT_ID", "\"$bannerAdUnitId\"")
            buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", "\"$nativeMoreAdUnitId\"")
            buildConfigField("String", "FCM_TOPIC", "\"$fcmTopic\"")
            buildConfigField("String", "MOBILE_API_TOKEN", "\"$mobileApiToken\"")
            buildConfigField("boolean", "ALLOW_INSECURE_WEB_CONTENT", "true")
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }

        release {
            isMinifyEnabled = false
            buildConfigField("String", "BASE_WEB_URL", "\"$releaseBaseWebUrl\"")
            buildConfigField("String", "BANNER_AD_UNIT_ID", "\"$bannerAdUnitId\"")
            buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", "\"$nativeMoreAdUnitId\"")
            buildConfigField("String", "FCM_TOPIC", "\"$fcmTopic\"")
            buildConfigField("String", "MOBILE_API_TOKEN", "\"$mobileApiToken\"")
            buildConfigField("boolean", "ALLOW_INSECURE_WEB_CONTENT", "false")
            manifestPlaceholders["usesCleartextTraffic"] = "false"
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    buildFeatures {
        buildConfig = true
        viewBinding = true
    }
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation("androidx.core:core-ktx:1.18.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.13.0")
    implementation("com.google.android.material:material:1.13.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.2.0")

    implementation(platform("com.google.firebase:firebase-bom:34.11.0"))
    implementation("com.google.firebase:firebase-analytics")
    implementation("com.google.firebase:firebase-messaging")
    implementation("com.google.firebase:firebase-crashlytics")

    implementation("com.google.android.gms:play-services-ads:25.1.0")
}
