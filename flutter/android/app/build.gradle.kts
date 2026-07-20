import java.util.Properties

plugins {
    id("com.android.application")
    id("com.google.gms.google-services")
    id("com.google.firebase.crashlytics")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

fun loadProperties(file: File): Properties =
    Properties().apply {
        if (file.exists()) {
            file.inputStream().use(::load)
        }
    }

val flutterLocalProperties = loadProperties(rootProject.file("local.properties"))
val legacyLocalProperties = loadProperties(rootProject.file("../../android/local.properties"))
val signingPropertiesFile =
    listOf(rootProject.file("key.properties"), rootProject.file("../../android/key.properties"))
        .firstOrNull(File::exists)
val signingProperties = signingPropertiesFile?.let(::loadProperties) ?: Properties()

fun localProperty(name: String): String? =
    sequenceOf(
        providers.gradleProperty(name).orNull,
        flutterLocalProperties.getProperty(name),
        legacyLocalProperties.getProperty(name),
    ).mapNotNull { it?.trim()?.ifBlank { null } }.firstOrNull()

fun quoted(value: String): String = "\"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\""

val releaseBaseWebUrl = "https://www.ikaimo-ai.com/keiba"
val googleTestAdmobAppId = "ca-app-pub-3940256099942544~3347511713"
val googleTestBannerAdUnitId = "ca-app-pub-3940256099942544/9214589741"
val googleTestNativeAdUnitId = "ca-app-pub-3940256099942544/2247696110"
val googleTestRewardedAdUnitId = "ca-app-pub-3940256099942544/5224354917"
val productionCustomPredictionRewardedAdUnitId = "ca-app-pub-2154555054261960/3052858105"
val admobAppId = localProperty("KEIBA_ADMOB_APP_ID") ?: googleTestAdmobAppId
val debugBaseWebUrl = localProperty("KEIBA_DEBUG_BASE_WEB_URL") ?: "http://10.0.2.2:8000/keiba"
val bannerAdUnitId = localProperty("KEIBA_BANNER_AD_UNIT_ID") ?: googleTestBannerAdUnitId
val nativeMoreAdUnitId = localProperty("KEIBA_NATIVE_MORE_AD_UNIT_ID") ?: googleTestNativeAdUnitId
val customPredictionRewardedAdUnitId =
    localProperty("KEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID")
        ?: productionCustomPredictionRewardedAdUnitId
val fcmTopic = localProperty("KEIBA_FCM_TOPIC") ?: "keiba-public-updates"
val fcmCentralTopic = localProperty("KEIBA_FCM_CENTRAL_TOPIC") ?: "$fcmTopic-central"
val fcmLocalTopic = localProperty("KEIBA_FCM_LOCAL_TOPIC") ?: "$fcmTopic-local"

val validateReleaseConfiguration =
    tasks.register("validateReleaseConfiguration") {
        group = "verification"
        description = "Validates production AdMob IDs and the existing Google Play signing key."
        doLast {
            val problems = mutableListOf<String>()
            if (admobAppId == googleTestAdmobAppId) problems += "KEIBA_ADMOB_APP_ID is missing."
            if (bannerAdUnitId == googleTestBannerAdUnitId) problems += "KEIBA_BANNER_AD_UNIT_ID is missing."
            if (nativeMoreAdUnitId == googleTestNativeAdUnitId) problems += "KEIBA_NATIVE_MORE_AD_UNIT_ID is missing."
            if (customPredictionRewardedAdUnitId == googleTestRewardedAdUnitId) {
                problems += "KEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID is missing."
            }
            if (signingPropertiesFile == null) problems += "key.properties for the existing Play signing key is missing."
            if (problems.isNotEmpty()) throw GradleException(problems.joinToString("\n"))
        }
    }

tasks.configureEach {
    if (name.contains("Release", ignoreCase = true) && name != validateReleaseConfiguration.name) {
        dependsOn(validateReleaseConfiguration)
    }
}

android {
    namespace = "com.ikaimo.keiba.app"
    compileSdk = 37
    ndkVersion = flutter.ndkVersion

    compileOptions {
        isCoreLibraryDesugaringEnabled = true
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.ikaimo.keiba.app"
        minSdk = 26
        targetSdk = 37
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        buildConfigField("String", "BASE_WEB_URL", quoted(releaseBaseWebUrl))
        buildConfigField("String", "BANNER_AD_UNIT_ID", quoted(bannerAdUnitId))
        buildConfigField("String", "NATIVE_MORE_AD_UNIT_ID", quoted(nativeMoreAdUnitId))
        buildConfigField(
            "String",
            "CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID",
            quoted(customPredictionRewardedAdUnitId),
        )
        buildConfigField("String", "FCM_TOPIC", quoted(fcmTopic))
        buildConfigField("String", "FCM_CENTRAL_TOPIC", quoted(fcmCentralTopic))
        buildConfigField("String", "FCM_LOCAL_TOPIC", quoted(fcmLocalTopic))
        manifestPlaceholders["usesCleartextTraffic"] = "false"
        manifestPlaceholders["admobAppId"] = admobAppId
    }

    signingConfigs {
        if (signingPropertiesFile != null) {
            create("release") {
                keyAlias = signingProperties.getProperty("keyAlias")
                keyPassword = signingProperties.getProperty("keyPassword")
                storeFile = signingProperties.getProperty("storeFile")?.let(::file)
                storePassword = signingProperties.getProperty("storePassword")
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-flutter-debug"
            buildConfigField("String", "BASE_WEB_URL", quoted(debugBaseWebUrl))
            buildConfigField(
                "String",
                "BANNER_AD_UNIT_ID",
                quoted(googleTestBannerAdUnitId),
            )
            buildConfigField(
                "String",
                "NATIVE_MORE_AD_UNIT_ID",
                quoted(googleTestNativeAdUnitId),
            )
            buildConfigField(
                "String",
                "CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID",
                quoted(googleTestRewardedAdUnitId),
            )
            manifestPlaceholders["admobAppId"] = googleTestAdmobAppId
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }
        release {
            if (signingPropertiesFile != null) {
                signingConfig = signingConfigs.getByName("release")
            }
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    buildFeatures {
        buildConfig = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")
    implementation("com.google.android.material:material:1.14.0")
}
