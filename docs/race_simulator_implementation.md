# 模拟器本地首版实施记录

日期：2026-11-01。分支：`codex/race-simulator`。

已完成本地代码接入和自动验证，尚未部署到生产或发布商店更新包。默认模型仍为 `illustrative-terrain-v2`；没有将研究模型切成正式模型，也没有重新调参。

## 已实装的链路

|部分|本地实现|
|---|---|
|输入发布|采集器保存有效 `RaceData` 后发布白名单输入；远端预测任务可以在 LLM 完成前回传输入；完整审计为兼容来源|
|独立采集|`scripts/collect_simulator_inputs.py` 发现当天中央、地方赛程，以最多3个线程收集出马表和近走，不调用 LLM；失败比赛下次重试|
|版本保存|`pipeline/data/simulator_inputs/{date}/{race_id}/` 保存不可变内容版本和最新索引；获取时间不单独改变内容哈希；历史审计不能替换较新的采集版本|
|数据隔离|只返回比赛、马匹和更早日期的近走白名单；取消、除外马不参赛；不返回行情、目标赛果、预测分数、审计记忆和内部路径|
|历史时钟|只关联指定的历史比赛；用已有 Agent 结果与历史身份资料核对日期、距离、身份和名次；信息不足或歧义时保留缺失|
|公开接口|模拟列表、指定比赛、固定输入版本、模型配置，支持 ETag；准备中、未找到、缺失版本、对象不支持分别处理|
|网页|详情页进入视口后加载真实单场数据；完整默认播放、暂停、同种子重播、模拟入线顺位、数据版本提示和 App 入口|
|共享渲染|网页和 App 的嵌入页使用同一个引擎及赛道组件；200次统计使用可终止的 Web Worker，旧任务不能覆盖新条件|
|Flutter 导航|保持5个底部标签，第3个替换为 `シミュレーション`；独立选赛页包含日本日期、中央/地方、场地和发走予定筛选；旧日报通知通过独立网页打开|
|Flutter 操作|播放、暂停、拖动、倍速、马群/赛道视角、步速、波动、不同种子、试验场地/距离、马匹估计状态、200次统计、比较集计和本地设置保存|
|状态保留|切标签保留当前比赛与状态，并暂停动画、取消未完成计算；个人倍速/视角和单场条件分开保存；网页带入的版本、模型和种子优先|
|链接|严格校验同站点模拟路径与参数；Flutter Router 接收冷启动和运行中链接；WebView 的普通详情入口进入同一模拟标签|
|下载承接|关联入口 `/keiba/open/simulator/{id}` 保留日期、输入版本、模型和种子；Android 商店链接核对为现有短链的实际 Play 地址；未配置 iOS 地址时如实显示下载案内准备中|
|记录与评价|默认单次和200次统计可保存，记录输入引用、模型、种子、代码提交和引擎摘要；独立评价脚本只读取 Agent 赛果，不改写输入或选择参数|
|埋点|网页预览、播放、完成、App入口、商店点击；App标签、选赛、成功打开、首次完成、设置变更和统计完成；分析失败不影响模拟|

## 本地查看

```powershell
# 构建网页（开发服务器也可用，API代理到本机8000端口）
cd frontend
npm run build
cd ../pipeline
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

示例地址：`http://127.0.0.1:8000/keiba/simulator?race_id=202605041101&date=2026-11-01`。本次用于联调的输入来自已保存的实际样本，不能据此声称当天所有比赛已经完成采集。

开发环境的原实验页通过 `/keiba/simulator?lab=1` 访问。生产构建的该参数不会显示高级网页面板。`app=1` 只作布局标记；完整控制面板属于 Flutter。

Android调试包：`flutter/build/app/outputs/flutter-apk/app-debug.apk`。默认调试连接 Android 模拟器宿主的 `10.0.2.2:8000`；连接真机时使用既有 `KEIBA_BASE_WEB_URL` 覆盖为可访问的开发地址。

## 采集、默认记录与评价命令

```powershell
python scripts/collect_simulator_inputs.py --date 2026-11-01 --workers 2 --force-refresh
node scripts/record_simulator_defaults.mjs --date 2026-11-01
python scripts/evaluate_simulator_defaults.py --date 2026-11-01
```

不联网导入本机已有 Agent 输入可使用 `--import-saved`。新模块只写模拟专用目录；不清理原始预测、赛果、审计包或 `analysis_outputs/` 快照。

默认记录保存在 `pipeline/data/simulator_runs/`。评价按中央/地方分别输出 Top5 对实际前三覆盖、全覆盖场次、至少两匹场次、单次覆盖、冠军保护指标、可用的用时误差和模型频率 Brier 误差。无实测末段标签时不可评价末段。当前未取得可核对的新赛果，评价脚本没有产生新的精度结论。

采集完成时刻和计划发走时刻不能证明实际发走前冻结，记录保留 `source_timestamp_unverified`。本机历史样本导入的获取时刻标为未知，不将导入时间冒充原始获取时间。

## 发布配置与仍需验收的事项

|配置|含义与默认值|
|---|---|
|`SIMULATOR_ENABLED`|本机默认开启；Render 环境默认关闭。关闭隐藏详情入口并停用公开模拟数据接口，保留已存数据|
|GitHub变量 `SIMULATOR_COLLECTION_ENABLED`|只有设为 `true` 才启用定时采集；新增工作流尚未推送或触发|
|`SIMULATOR_COLLECTION_CALLBACK_URL`|独立采集回传地址：`https://www.ikaimo-ai.com/keiba/internal/simulator-input`|
|`PIPELINE_CALLBACK_SECRET`|沿用已有 HMAC 密钥，放在服务环境和 GitHub Secrets；不提交密钥|
|`SIMULATOR_ANDROID_STORE_URL`|已核对默认值为 `https://play.google.com/store/apps/details?id=com.ikaimo.keiba.app`|
|`SIMULATOR_IOS_STORE_URL`|真实 App Store 地址，当前未配置，不填虚构地址|
|`SIMULATOR_ANDROID_SHA256`|Google Play 正式签名证书公开指纹，多个用逗号分隔；缺少有效值时关联文件返回503|
|`SIMULATOR_IOS_TEAM_ID`|正式签名 Team ID，必须核对后配置；缺少有效值时关联文件返回503|

Android Manifest 和 iOS Associated Domains 只覆盖 `www.ikaimo-ai.com/keiba/open/simulator/`。两个关联文件由站点根路径 `/.well-known/` 返回，不能只部署 `/keiba` 静态资源。需要实际部署、正式签名安装和系统验证；本机有文件或调试包编译通过不能代表已验证关联。依据：[Android App Links 验证](https://developer.android.com/training/app-links/verify-applinks)、[Flutter 深层链接](https://docs.flutter.dev/ui/navigation/deep-linking)。

仍需完成：Android/iPhone 真机的动画、桥接、后台恢复与冷暖启动测试；iOS Mac/Xcode 构建及签名；Safari 同域关联入口、常见内置浏览器和旧 Android 版本回退；正式包身份确认；普通设备上的帧率/耗时/内存基线；小范围开启后的事件去重和采集覆盖检查。未安装用户首次启动自动接续仍属于后续范围。

## 本次验证

- Python：49项，47项通过、2项原有条件性跳过；覆盖新输入/API、公开赛程缓存与既有解析、采集和预测阶段。
- JavaScript：2项通过，覆盖上下文校验、取消马排除、固定种子复现及无动画统计与完整轨迹的终点一致。
- Flutter：37项通过，覆盖已有功能、模拟链接、单场参数隔离、发走时间未知及地方准备中比赛的筛选。
- 前端生产构建通过；Flutter 静态分析无问题；Android调试包构建通过。
- 浏览器确认真实单场马匹、赛道、完整播放和资料获取时刻未知的提示；网页未开放 Flutter 高级设置。
- 日语按 `natural-japanese` 的人工清单检查。本机缺少 `uv` 与 Sudachi，lint/calques 的自动运行未通过依赖加载；未把人工检查描述为自动检查通过。将波动标签写为 `スタート・調子のばらつき`，保留 `シミュレーション`、`My予測` 和既有术语。

生产开放和商店发布没有在本次执行。后续按原计划第4—6阶段，补齐正式配置及双端真机验收后分批开启。
