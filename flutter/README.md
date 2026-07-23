# いかいもAI競馬 Flutter クライアント

既存の Android 公開版を Flutter へ置き換え、Android / iOS の両方で同じ機能を提供するクライアントです。比較・切り戻し用として `../android/` は維持し、このディレクトリを共通実装として使用します。

## Android / iOS 共通の実装範囲

- `レース / 履歴分析 / 日報 / My予測 / その他` の下部ナビゲーション
- Agent の評価要素をレース内で標準化し、保存した相対バランスで端末内計算するMy予測（点数は非表示）
- `app=1` を維持するアプリ内 URL 正規化
- 読み込み進捗、再試行、再読み込み、先頭からのプル更新
- サイト内リンク、外部リンク、ダウンロードの振り分け
- JavaScript、DOM Storage、Cookie、公開版 User-Agent（Android は第三者 Cookie も明示的に許可）
- Firebase Messaging、Android 通知チャンネル、iOS APNs、初回権限確認
- 旧 Android 版の通知設定を引き継ぐ互換ストレージ
- 通知トピック同期、フォアグラウンド／バックグラウンド表示、タップ遷移
- Firebase Analytics と Crashlytics
- メイン画面のアダプティブバナーと `その他` 画面のネイティブ広告
- Google UMP による広告同意フォーム、広告リクエスト制御、プライバシー設定入口
- 公開版と同じアプリアイコン、アプリ名、Firebase／AdMob 設定

## 識別子とバージョン

- Android リリース applicationId: `com.ikaimo.keiba.app`
- Android デバッグ applicationId: `com.ikaimo.keiba.app.debug`
- iOS bundle ID: `com.ikaimo.keiba.app`
- バージョン: `1.3.0 (4)`
- Android minSdk: 26、targetSdk / compileSdk: 37
- iOS deployment target: 15.0

Google Play の既存アプリを更新するため、リリース版は旧 Android 版と同じ署名鍵が必要です。`key.properties` と署名鍵がない状態では、リリース構成の検証が失敗するようにしています。デバッグ署名で正式版を上書きすることはありません。

## ローカル設定

Firebase 設定ファイルと署名ファイルは Git 管理対象外です。

- `android/app/google-services.json`
- `android/app/src/debug/google-services.json`
- `android/key.properties`
- `*.jks` / `*.keystore`
- `ios/Runner/GoogleService-Info.plist`
- `ios/Flutter/Local.xcconfig`

Android の AdMob ID はリポジトリ直下の既存 `../android/local.properties` を優先して読み込みます。iOS は `ios/Flutter/Local.xcconfig.example` を `Local.xcconfig` にコピーし、iOS 用の本番 AdMob app ID、バナー ID、ネイティブ広告 ID を設定します。未設定のデバッグ環境では各 OS 用の Google テスト ID を使い、リリース検証ではテスト ID を拒否します。

### iOS の Firebase / Push 設定

1. Firebase プロジェクトへ bundle ID `com.ikaimo.keiba.app` の Apple アプリを登録します。
2. ダウンロードした `GoogleService-Info.plist` を `ios/Runner/` に配置します。
3. Firebase Console の Cloud Messaging へ Apple Developer の APNs 認証キー（`.p8`）を登録します。
4. Xcode で `ios/Runner.xcworkspace` を開き、Runner の Signing Team を選択します。Push Notifications と Background Modes はプロジェクトに設定済みです。
5. Crashlytics の dSYM 自動アップロードを有効にするため、Firebase CLI へログイン後に `flutterfire configure` を一度実行します。

### AdMob のプライバシーメッセージ

AdMob の「プライバシーとメッセージ」で配信地域に必要なメッセージを公開します。アプリは起動時に毎回 UMP の同意情報を更新し、必要なフォームが完了して `canRequestAds` が有効になるまで広告を読み込みません。Google が継続的なプライバシー設定入口を必須と判定した場合は、`その他` 画面に「広告のプライバシー設定」を表示します。

現在の iOS 実装は IDFA / ATT を要求しません。AdMob 側で IDFA メッセージを有効にする場合は、先に `NSUserTrackingUsageDescription` と ATT の実装を追加し、App Store Connect のプライバシー回答も実際の動作に合わせて更新してください。

### iOS の My予測リワード制御

Firebase Remote Config の Boolean パラメータ `iOSRewardAD` で、iOS の My予測にリワード広告を必須とするかを制御します。`false` の場合は広告なしで利用でき、`true` の場合は動画を最後まで視聴すると1時間利用できます。アプリ内の既定値および Remote Config の取得失敗時は `false` です。Android はこのパラメータを参照せず、従来どおりリワード広告を必須とします。

## ローカル実行

先にリポジトリ直下で Web サーバーを起動します。

```shell
python pipeline/web_server.py
```

別のターミナルで実行します。

```shell
cd flutter
flutter run
```

デバッグ時の既定 URL は Android エミュレーターで `http://10.0.2.2:8000/keiba`、iOS Simulator で `http://localhost:8000/keiba` です。公開サイトへ接続する場合は次のように上書きできます。

```shell
flutter run --dart-define=KEIBA_BASE_WEB_URL=https://www.ikaimo-ai.com/keiba
```

## 検証

```shell
flutter analyze
flutter test
flutter build apk --debug
flutter build ios --debug --no-codesign
cd android
./gradlew :app:validateReleaseConfiguration
```

実機では通知権限、FCM の三つのトピック、通知タップ、バナー広告、ネイティブ広告、外部リンク、ダウンロード、WebView のログイン Cookie を両 OS で確認します。iOS の APNs/FCM は Simulator ではなく実機を使い、前面・背面・終了状態からの表示とタップ遷移を最終確認します。
