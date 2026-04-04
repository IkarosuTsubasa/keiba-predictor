plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.gms.google-services")
    id("com.google.firebase.crashlytics")
}

val releaseBaseWebUrl = "https://www.ikaimo-ai.com/keiba"
val admobAppId =
    providers.gradleProperty("KEIBA_ADMOB_APP_ID")
        .orElse("ca-app-pub-3940256099942544~3347511713")
        .get()
val debugBaseWebUrl =
    providers.gradleProperty("KEIBA_DEBUG_BASE_WEB_URL")
        .orElse("http://10.0.2.2:8000/keiba")
        .get()
val bannerAdUnitId =
    providers.gradleProperty("KEIBA_BANNER_AD_UNIT_ID")
        .orElse("ca-app-pub-3940256099942544/9214589741")
        .get()
val nativeMoreAdUnitId =
    providers.gradleProperty("KEIBA_NATIVE_MORE_AD_UNIT_ID")
        .orElse("ca-app-pub-3940256099942544/2247696110")
        .get()

android {
    namespace = "com.ikaimo.keiba.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ikaimo.keiba.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "BASE_WEB_URL", "\"$releaseBaseWebUrl\"")
        buildConfigField("String", "BANNER_AD_UNIT_ID", "\"$bannerAdUnitId\"")
        buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", "\"$nativeMoreAdUnitId\"")
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
            buildConfigField("boolean", "ALLOW_INSECURE_WEB_CONTENT", "true")
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }

        release {
            isMinifyEnabled = false
            buildConfigField("String", "BASE_WEB_URL", "\"$releaseBaseWebUrl\"")
            buildConfigField("String", "BANNER_AD_UNIT_ID", "\"$bannerAdUnitId\"")
            buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", "\"$nativeMoreAdUnitId\"")
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
