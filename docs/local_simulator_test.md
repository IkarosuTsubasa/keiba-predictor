# 本地模拟测试入口

已准备的真实样本：2026-11-01 京都11R「まほろばS」，芝1600m，13匹马、268条目标比赛之前的近走。比赛ID为 `202608041111`。

天气、马场状态和内外回标记缺失，页面保留部分资料提示。此次拉取用于界面与功能联调，不作为赛前预测验证。

## 电脑浏览器

打开：<http://127.0.0.1:8000/keiba/simulator?race_id=202608041111&date=2026-11-01>

本地后端目前运行在8000端口；关闭后可重新启动：

```powershell
Set-Location D:/keiba/new/pipeline
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## Android 模拟器

本次建立了独立的 `KeibaLocalTest`（窗口标题 `Keiba Local Test`）测试模拟器，并安装了本地版 App。原 Pixel_10 空间不足，未删除其数据；已连接的实体手机未安装或修改。

本地版包名为 `com.ikaimo.keiba.app.debug`，固定连接 `http://10.0.2.2:8000/keiba`。这是 Android 模拟器访问电脑本机服务的地址。正式包继续连接线上站点。

操作步骤：

1. 打开 App 底部第3个标签 `シミュレーション`。
2. 日期选择 `2026-11-01`，关闭 `発走予定のレースのみ`。该场计划发走时间已过，需要查看当天全部比赛。
3. 选择 `京都 11R · まほろばS`，点击 `再生`。
4. 向下滚动可调整倍速、视角、步速、波动和试验条件，也可运行200次统计。

模拟器专用安装包：`D:/keiba/new/flutter/build/app/outputs/flutter-apk/ikaimo-keiba-local-emulator.apk`（x86_64，约73.7MiB）。该文件用于本地模拟器，不替换连接线上的正式包。

模拟器注册信息在本机 Android AVD 目录中，独立数据目录为 `D:/keiba/new/output/android-avd/KeibaLocalTest.avd`。可通过 Android Studio 的设备管理器重新打开。

## 本次检查

本机列表和单场接口均已返回该场真实输入；浏览器与 Android 模拟器都已加载赛道，Android 原生播放按钮已驱动马匹动画。原始 Agent 预测、赛果与审计数据未改写；本次输入及默认试跑记录写入模拟专用目录。
