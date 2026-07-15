# いかいもAI競馬 Flutter クライアント

既存の Android 公開版を Flutter へ置き換えるためのクライアントです。移行完了までは比較・切り戻し用として `../android/` を維持し、このディレクトリだけで新実装を進めます。iOS 対応は Android 移行後に行います。

## Android 実装範囲

- `レース / 履歴分析 / 日報 / その他` の下部ナビゲーション
- `app=1` を維持するアプリ内 URL 正規化
- 読み込み進捗、再試行、再読み込み、先頭からのプル更新
- サイト内リンク、外部リンク、ダウンロードの振り分け
- JavaScript、DOM Storage、Cookie、第三者 Cookie、公開版 User-Agent
- Firebase Messaging、通知チャンネル、初回権限確認
- 旧 Android 版の通知設定を引き継ぐ互換ストレージ
- 通知トピック同期、フォアグラウンド／バックグラウンド表示、タップ遷移
- Firebase Analytics と Crashlytics
- メイン画面のアダプティブバナーと `その他` 画面のネイティブ広告
- 公開版と同じアプリアイコン、パッケージ名、Firebase／AdMob 設定

## 識別子とバージョン

- リリース applicationId: `com.ikaimo.keiba.app`
- デバッグ applicationId: `com.ikaimo.keiba.app.debug`
- バージョン: `1.3.0 (4)`
- minSdk: 26
- targetSdk / compileSdk: 37

Google Play の既存アプリを更新するため、リリース版は旧 Android 版と同じ署名鍵が必要です。`key.properties` と署名鍵がない状態では、リリース構成の検証が失敗するようにしています。デバッグ署名で正式版を上書きすることはありません。

## ローカル設定

Firebase 設定ファイルと署名ファイルは Git 管理対象外です。

- `android/app/google-services.json`
- `android/app/src/debug/google-services.json`
- `android/key.properties`
- `*.jks` / `*.keystore`

AdMob ID はリポジトリ直下の既存 `../android/local.properties` を優先して読み込みます。未設定のデバッグ環境では Google のテスト ID を使い、リリース検証ではテスト ID を拒否します。

## ローカル実行

先にリポジトリ直下で Web サーバーを起動します。

```powershell
python pipeline/web_server.py
```

別のターミナルで実行します。

```powershell
cd flutter
flutter run
```

デバッグ時の既定 URL は Android エミュレーター用の `http://10.0.2.2:8000/keiba` です。公開サイトへ接続する場合は次のように上書きできます。

```powershell
flutter run --dart-define=KEIBA_BASE_WEB_URL=https://www.ikaimo-ai.com/keiba
```

## 検証

```powershell
flutter analyze
flutter test
flutter build apk --debug
cd android
gradlew.bat :app:validateReleaseConfiguration
```

実機では通知権限、FCM の三つのトピック、通知タップ、バナー広告、ネイティブ広告、外部リンク、ダウンロード、WebView のログイン Cookie を確認します。
