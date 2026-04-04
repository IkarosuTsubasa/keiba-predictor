plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val releaseBaseWebUrl = "https://www.ikaimo-ai.com/keiba"
val debugBaseWebUrl =
    providers.gradleProperty("KEIBA_DEBUG_BASE_WEB_URL")
        .orElse("http://10.0.2.2:8000/keiba")
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
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            buildConfigField("String", "BASE_WEB_URL", "\"$debugBaseWebUrl\"")
        }

        release {
            isMinifyEnabled = false
            buildConfigField("String", "BASE_WEB_URL", "\"$releaseBaseWebUrl\"")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = true
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.18.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.2.0")
}
