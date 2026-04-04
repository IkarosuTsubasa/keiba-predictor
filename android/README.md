# Android App Shell

这个目录是当前项目的 Android 原生壳工程预览版，定位是：

- 原生 `Kotlin` 壳子
- `WebView` 承载公开页面
- 底部导航直达 `レース / 履歴分析 / 私の日報 / その他`

## 当前入口

- `レース`: `/keiba`
- `履歴分析`: `/keiba/history`
- `私の日報`: `/keiba/reports`
- `その他`: `プライバシーポリシー / 利用規約 / 免責事項 / お問い合わせ`

## 目录

```text
android/
  app/
    src/main/
      java/com/ikaimo/keiba/app/
      res/
```

## 打开方式

1. 用 Android Studio 打开 `D:\keiba\new\android`
2. 等待 Gradle Sync
3. 运行 `app`

## Debug 本地检查

如果你想在本地检查 Web 改动：

1. 在仓库根目录启动本地服务：
   `python pipeline/web_server.py`
2. 在 Android Studio 运行 `debug` 版本
3. `debug` 默认连接：
   `http://127.0.0.1:8000/kei`

说明：

- `127.0.0.1` 是 Android 模拟器访问宿主机 localhost 的固定地址
- `release` 版本仍然连接线上：
  `https://www.ikaimo-ai.com/keiba`

## 可选覆盖

如果你不是用模拟器，想让 `debug` 改连别的地址，可以加一个 Gradle 属性：

`KEIBA_DEBUG_BASE_WEB_URL=http://你的地址:8000/keiba`

这个属性可以放在：

- `C:\Users\你的用户名\.gradle\gradle.properties`
- 或者项目里的 `android\gradle.properties`
