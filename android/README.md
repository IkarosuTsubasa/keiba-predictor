# Android App Shell

This directory contains the preview version of the native Android shell for the current project.

- Native `Kotlin` shell
- Public pages rendered inside `WebView`
- Bottom navigation for `レース / 履歴分析 / 日報 / その他`
- `その他` uses a native screen for entry points and shows a `Native Advanced` ad at the bottom

## Current Entry Points

- `レース`: `/keiba`
- `履歴分析`: `/keiba/history`
- `日報`: `/keiba/reports`
- `その他`: `プライバシーポリシー / 利用規約 / 免責事項 / お問い合わせ`

## Directory Layout

```text
android/
  app/
    src/main/
      java/com/ikaimo/keiba/app/
      res/
```

## Open the Project

1. Open `D:\keiba\new\android` in Android Studio
2. Wait for Gradle sync to finish
3. Run the `app` module

## Local Debug Workflow

If you want to verify local Web changes from the Android app:

1. Start the local server from the repository root:
   `python pipeline/web_server.py`
2. Run the `debug` build in Android Studio
3. The `debug` build connects to:
   `http://10.0.2.2:8000/keiba`

Notes:

- `10.0.2.2` is the fixed host alias used by the Android Emulator to access the host machine's localhost
- The `release` build still points to:
  `https://www.ikaimo-ai.com/keiba`
- The bottom banner ad uses the configured AdMob banner unit

## Optional Overrides

If you are not using the Android Emulator and want `debug` to connect to a different URL, add this Gradle property:

`KEIBA_DEBUG_BASE_WEB_URL=http://your-host:8000/keiba`

You can place it in:

- `C:\Users\<your-user>\.gradle\gradle.properties`
- or `android\gradle.properties`

The repository does not include `google-services.json`.
Add your own local files here before building Firebase-enabled variants:

- `android/app/google-services.json`
- `android/app/src/debug/google-services.json`

Local machine-only values can be placed either in Gradle properties or in `android/local.properties`.
Project-level `android/local.properties` is already ignored by Git.

To override the AdMob app ID:

- `KEIBA_ADMOB_APP_ID=your_admob_app_id`

To override the banner ad unit:

- `KEIBA_BANNER_AD_UNIT_ID=your_banner_id`

To override the native ad unit shown on the `その他` screen:

- `KEIBA_NATIVE_MORE_AD_UNIT_ID=your_native_id`

Release builds fail on purpose if these AdMob values are missing or still set to Google's test IDs.

To override the FCM topic used by the Android client:

- `KEIBA_FCM_TOPIC=your_topic_name`

## FCM

The Android shell already includes the client-side FCM integration:

- `FirebaseMessagingService` for receiving messages
- Notification permission request on first launch for Android 13+
- Automatic notification channel creation
- Notification tap routing to a target page

It is recommended to send a `data` payload with these fields:

- `title`: notification title
- `body`: notification body
- `destination`: top-level destination, one of `races / history / reports / more`
- `url`: relative in-app path or full URL, for example `/keiba/reports/foo` or `https://...`

Priority:

- If both `url` and `destination` are present, `url` takes priority
- `app=1` is appended automatically for routed Web pages
