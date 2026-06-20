# Keiba LLM Agent MVP

这个目录实现了第一阶段本地最小闭环：

1. 读取 `race_data.json`
2. 读取 `memory/lessons.json`
3. 生成结构化 `prediction.json`
4. 读取 `result.json`
5. 生成 `review.json`
6. 从 `review.json` 抽取 lessons 并追加到 `memory/lessons.json`

当前版本默认使用真实 LLM provider；支持通过 Gemini API 或 OpenAI API 做 summary / review 文案增强，缺少 key 且 fallback 开启时会回退到 `MockLLMClient`。
所有 LLM 输出都强制为 JSON；没有提供的数据在 prompt 中要求写成 `unknown`。
当前 analysis 有两种模式：
- 没有 `recent_runs`：走 Mock/LLM fallback
- 有 `recent_runs`：优先走 heuristic recent-run scorer
当前 scorer 已将短期状态与长期能力拆分：`recent_form` 保留为旧字段兼容，但实际表示近3走表现质量；`ability_score` 使用全生涯有效成绩的上位表现与生涯均值混合；距離・コース・馬場适性使用全生涯 recent_runs 统计，避免只看近5走导致中央赛马或条件转换马被低估。
当前还启用了 `Race Deep Analyzer v1`，会基于现有 `race_data / recent_runs / odds / popularity / jockey` 为每匹马生成更细的结构化分析，用于增强解释性、report 和 social 文案。
当前还启用了 `Race Level / Opponent Analysis v1`，会基于 `recent_runs` 中的 `race_id / field_size / popularity / finish`，分析本场再战关系与轻量レースレベル，并作为基础分项与轻量补正共同接入 `total_score`。
当前还启用了 `Pace / Running Style Analyzer v1`，会基于 `recent_runs` 中可解析的 `通過順 / 上り` 推断脚质和本场展開。取不到时会按 `unknown` 处理，第一版仍是轻量推定，并作为展开展・騎手分项与小幅补正接入评分。
当前还启用了 `Pedigree Analyzer v1.2`，会从 netkeiba horse cache / pedigree cache 解析父・母・母父，并基于父・母父自身的実績 profile 生成轻量血統分析。
当前还启用了 `Pedigree Score Integration v1`，会将血統分析以轻量 bonus / penalty 的方式反映到 `horse_scores.total_score`。该补正最大 `+2.0`、最小 `-1.5`，仍然只是 heuristic 的解释性补正，不是正式学习模型。
`review-url` 生成的 lessons 会在后续 `analysis / analyze-url` 中按相似条件检索并写入 `prediction.used_lessons`。
当前 lesson 只做轻量加权与说明，不是模型训练。
`RaceData.horses[].odds / popularity` 会优先解析当前出馬表中的本场値；如果页面源码没有提供，则保持 `null` 并回退到 `recent_runs` 的历史赔率信息。
`prediction.strategy` 会输出当前的买入判断，包括：
- `bet_decision`
- `confidence`
- `participation_level`
- `reason_codes`
- `reason`

`prediction.deep_analyses` 会保存每匹马的：
- `positive_flags`
- `risk_flags`
- `recent_form_summary`
- `distance_analysis`
- `course_analysis`
- `track_condition_analysis`
- `jockey_analysis`
- `odds_analysis`
- `overall_comment`

`bets` 会根据 `strategy` 自动收缩或清空。当前仍是 heuristic policy，不是资金管理模型。

## 运行方式

在仓库根目录执行：

```bash
python keiba_llm_agent/main.py analysis --race-data keiba_llm_agent/data/samples/sample_race_data.json
```

默认会输出并保存到：

```text
keiba_llm_agent/data/predictions/sample_001.json
```

执行赛后 review：

```bash
python keiba_llm_agent/main.py review --race-id sample_001 --result keiba_llm_agent/data/samples/sample_result.json
```

默认会输出并保存到：

```text
keiba_llm_agent/data/reviews/sample_001.json
```

同时会把 `review.lessons` 追加保存到：

```text
keiba_llm_agent/memory/lessons.json
```

提取 netkeiba URL 中的 `race_id`：

```bash
python keiba_llm_agent/main.py parse-url --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511"
```

输出：

```text
race_id: 202605180511
```

解析本地保存的 netkeiba 出馬表 HTML：

```bash
python keiba_llm_agent/main.py parse-html --html tests/fixtures/netkeiba_shutuba_sample.html --race-id 202605180511
```

该命令会把解析后的 `RaceData` JSON 输出到 stdout。

抓取 netkeiba 出馬表 URL，缓存 HTML，并保存解析后的 `race_data.json`：

```bash
python keiba_llm_agent/main.py fetch-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511"
```

强制刷新缓存：

```bash
python keiba_llm_agent/main.py fetch-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511" --force-refresh
```

同时抓取每匹马全生涯成绩并写入 `recent_runs`：

```bash
python keiba_llm_agent/main.py fetch-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511" --with-recent-runs
```

该命令默认优先读取 `keiba_llm_agent/data/html_cache/` 中的缓存。
请低频、手动触发，不要批量高频访问 netkeiba。

从 netkeiba URL 直接生成 `race_data.json` 和 `prediction.json`：

```bash
python -m keiba_llm_agent.main analyze-url --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511"
```

只检查抓取与解析结果，不执行分析：

```bash
python -m keiba_llm_agent.main analyze-url --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511" --dry-run
```

强制刷新 HTML cache：

```bash
python -m keiba_llm_agent.main analyze-url --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511" --force-refresh
```

同时抓取每匹马近走后再做 analysis：

```bash
python -m keiba_llm_agent.main analyze-url --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511" --with-recent-runs
```

单独抓取一匹马页面并调试 `recent_runs` 解析：

```bash
python keiba_llm_agent/main.py fetch-horse --horse-id 2021104073
```

单独解析一匹马的父・母・母父：

```bash
python -m keiba_llm_agent.main parse-pedigree --horse-id 2021104073
```

Pedigree Analyzer v1.2 说明：
- 从 `horse_html_cache / pedigree_html_cache` 解析父 / 母 / 母父
- 父・母父的実績 profile 会作为 surface / distance / track condition / class power 的主判断材料
- 静态 sire knowledge table 已移除；取不到父・母父実績时按 `unknown` 处理
- 目前仍不是完整血統数据库，兄弟马特性尚未作为正式评分因子

Pedigree Score Integration v1：
- 血統補正只做轻量参与，不覆盖 `recent_runs` 主评分
- `horse_scores` 会保存 `base_total_score / pedigree_adjustment / total_score`
- 血統補正范围：最大 `+2.0`，最小 `-1.5`
- 后续会结合回顾结果继续微调

Race Level / Opponent Analysis v1：
- 检测本场马之间是否存在 recent_runs 的再战关系
- 基于 `field_size / popularity / finish` 做轻量レースレベル判断
- 会输出 `race_level_analyses`
- 第28阶段开始会以轻量补正接入 `total_score`
- race level 补正范围：最大 `+1.0`，最小 `-1.0`

Pace / Running Style Analyzer v1：
- 如果 `recent_runs` 中能解析到 `通過順 / 上り`，则会推断 `逃げ / 先行 / 差し / 追込`
- 会生成 `pace_analyses` 和 `race_pace_projection`
- 如果数据不足，则按 `unknown` 处理
- 第28阶段开始支持接入 `total_score`，当前默认权重已降为 `0.0`
- pace 补正范围：最大 `+0.8`，最小 `-0.8`

Analysis Adjustment Integration v1：
- `race_level_analyses` 与 `pace_analyses` 支持以轻量 adjustment 参与最终 `total_score`
- `horse_scores` 会保存 `race_level_adjustment / pace_adjustment / score_breakdown`
- `marks` 使用补正后的 `total_score` 重新排序
- 当前默认权重：`pedigree=0.2 / race_level=1.0 / pace=0.0`
- 仍然不是正式 ML 模型，后续会继续结合实战结果微调

抓取 netkeiba result URL，缓存结果页并保存 `result.json`：

```bash
python -m keiba_llm_agent.main fetch-result --url "https://race.netkeiba.com/race/result.html?race_id=202605020811"
```

根据 netkeiba result URL 自动生成 `review.json` 并追加 lessons：

```bash
python -m keiba_llm_agent.main review-url --url "https://race.netkeiba.com/race/result.html?race_id=202605020811"
```

根据 `prediction.json` 生成对外可读的 Markdown 预测报告：

```bash
python -m keiba_llm_agent.main report-prediction --race-id 202605020811
```

根据 `prediction.json` 和 `review.json` 生成 Markdown 回顾报告：

```bash
python -m keiba_llm_agent.main report-review --race-id 202605020811
```

根据 `prediction.json` 生成适合 X/Twitter 的赛前短文：

```bash
python -m keiba_llm_agent.main social-prediction --race-id 202605020811
```

根据 `prediction.json`、`review.json` 和可选 `result.json` 生成赛后短文：

```bash
python -m keiba_llm_agent.main social-review --race-id 202605020811
```

一键执行赛前完整流程：

```bash
python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605020811"
```

一键执行赛后完整流程：

```bash
python -m keiba_llm_agent.main review-race --url "https://race.netkeiba.com/race/result.html?race_id=202605020811"
```

检查 `race_data.json` 是否适合用于预测：

```bash
python -m keiba_llm_agent.main validate-race-data --race-id 202605020811
```

生成指定日期的日次汇总报告与社交短文：

```bash
python -m keiba_llm_agent.main daily-summary --date 2026-05-17
```

对已有 predictions / results / reviews 做区间 backtest 比较：

```bash
python -m keiba_llm_agent.main backtest --from-date 2026-05-01 --to-date 2026-05-31
```

只输出指定 mode，或限制最少样本数：

```bash
python -m keiba_llm_agent.main backtest --from-date 2026-05-01 --to-date 2026-05-31 --mode full_adjusted
python -m keiba_llm_agent.main backtest --from-date 2026-05-01 --to-date 2026-05-31 --min-races 5
```

对同一区间做 adjustment 权重 what-if 比较：

```bash
python -m keiba_llm_agent.main backtest-weights --from-date 2026-05-01 --to-date 2026-05-31
python -m keiba_llm_agent.main backtest-weights --from-date 2026-05-01 --to-date 2026-05-31 --pedigree-weight 0.5 --race-level-weight 0.5 --pace-weight 0.5
```

输出：
- `data/backtests/{from}_{to}_weight_tuning.json`
- `data/backtests/{from}_{to}_weight_tuning.md`

批量采集脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action predict -InputFile .\scripts\race_ids.example.txt
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action review -InputFile .\scripts\race_ids.example.txt
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action validate
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action backtest -FromDate 2026-05-01 -ToDate 2026-05-31
```

说明：
- `predict/review` 的 `-InputFile` 现在支持两种格式：
  - 完整 netkeiba URL
  - 仅 12 位 `race_id`
- 如果只写 `race_id`，脚本会自动补成 `shutuba` / `result` URL

常用参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action predict -InputFile .\my_race_ids.txt -SleepSeconds 5 -LlmProvider mock
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action predict -InputFile .\my_race_ids.txt -ForceRefresh -SkipSocial
powershell -ExecutionPolicy Bypass -File .\scripts\keiba_bulk_collect.ps1 -Action review -InputFile .\my_race_ids.txt -StopOnError
```

跳过 Markdown 报告或社交短文：

```bash
python -m keiba_llm_agent.main daily-summary --date 2026-05-17 --skip-report
python -m keiba_llm_agent.main daily-summary --date 2026-05-17 --skip-social
```

查看和管理 lesson memory：

```bash
python -m keiba_llm_agent.main lessons-list
python -m keiba_llm_agent.main lessons-disable --lesson-id lesson_xxx
python -m keiba_llm_agent.main lessons-enable --lesson-id lesson_xxx
python -m keiba_llm_agent.main lessons-prune --min-score 0.2
```

检查当前 LLM provider / model / fallback 状态：

```bash
python -m keiba_llm_agent.main llm-check
```

使用 OpenAI provider 执行赛前一键流程：

```bash
python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=202605020811" --llm-provider gemini
```

使用 OpenAI provider 执行赛后一键流程：

```bash
python -m keiba_llm_agent.main review-race --url "https://race.netkeiba.com/race/result.html?race_id=202605020811" --llm-provider gemini
```

默认会保存到：

```text
keiba_llm_agent/data/reports/{race_id}_prediction.md
keiba_llm_agent/data/reports/{race_id}_review.md
keiba_llm_agent/data/social_posts/{race_id}_prediction.txt
keiba_llm_agent/data/social_posts/{race_id}_review.txt
keiba_llm_agent/data/daily_reports/{date}.md
keiba_llm_agent/data/social_posts/{date}_daily.txt
```

`report-prediction / report-review` 会尽量读取以下文件来补全显示：
- `data/predictions/{race_id}.json`
- `data/race_data/{race_id}.json`
- `data/results/{race_id}.json`
- `data/reviews/{race_id}.json`

其中：
- prediction report 会优先使用 `race_data.json` 补充当前 `odds / popularity`
- prediction report 会输出上位5頭的 `深掘り分析`
- review report 会优先使用 `result.json` 显示实际 `1着 / 2着 / 3着`

## LLM 配置

环境变量：

- `KEIBA_LLM_PROVIDER=mock|gemini|openai`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `KEIBA_LLM_ENABLE_FALLBACK=true|false`

默认值：

- `KEIBA_LLM_PROVIDER=gemini`
- `GEMINI_MODEL=gemini-3.1-flash-lite`
- `OPENAI_MODEL=gpt-5.4-mini`
- `KEIBA_LLM_ENABLE_FALLBACK=true`

PowerShell 示例：

```powershell
$env:KEIBA_LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_api_key"
$env:GEMINI_MODEL="gemini-3.1-flash-lite"
python -m keiba_llm_agent.main llm-check
```

也可以直接在 `keiba_llm_agent/.env` 中配置上述变量。当前会自动优先读取系统环境变量；如果系统环境变量未设置，则回退读取 `keiba_llm_agent/.env`。

当前真实 LLM 只用于：

- prediction summary / risks / commentary 的语言增强
- review 的 good_points / bad_points / lessons 语言增强

不会直接修改：

- `horse_scores`
- `marks`
- `strategy`

`review-url` 会读取并依赖以下文件：
- `data/predictions/{race_id}.json`
- `data/results/{race_id}.json`
- `data/race_data/{race_id}.json`

lesson 的 `course / surface / distance / track_condition` 默认来自 `race_data.race_info`。
如果 `race_data.json` 缺失，则回退到 `prediction.json` 内附带的 `race_info`。
如果 `prediction.json` 里存在 `bets`，`review-url` 还会计算：
- `bet_results`
- `total_stake`
- `total_return`
- `roi`

所有涉及 horse / race fetch 的命令都应低频、手动触发，不要批量高频访问 netkeiba。
`recent_runs` 会自动排除当前目标比赛及未来比赛，避免 data leakage。
单独使用 `fetch-horse` 只用于调试 horse parser，不做目标比赛过滤。
赛前预测推荐使用 `analyze-url --with-recent-runs`。
`used_lessons` 会随 `prediction.json` 一起保存，便于确认本次分析参考了哪些过往复盘。
`odds_value` 会优先使用当前出馬表解析出的 `odds`，只有当前赔率缺失时才使用 `recent_runs` 的历史 `odds/popularity` 作为 fallback。
当前默认 `use_market_score_in_ranking = false`、`market_signal_weight = 0.0`，因此 `odds / popularity` 只保留为参考情報与误差分析字段，不直接进入 core ranking。
`Race Deep Analyzer v1` 只使用现有数据源：
- `race_data`
- `recent_runs`
- `odds / popularity`
- `jockey`

`Race Deep Analyzer v1` 本身不直接使用：
- 血统
- 调教
- Top3Prob

`Pedigree Analyzer v1.2` 会：
- 优先从本地 horse cache / pedigree cache 解析 `父 / 母 / 母父`
- 必要时从 `https://db.netkeiba.com/horse/ped/{horse_id}/` 做低频、缓存式 fallback
- 使用父・母父自身的実績 profile 做轻量分析

当前还不是本格的血统数据库，不明血统会按 `unknown` 处理。
`Race Simulation / LLM Reasoning v1` 会：
- 汇总 `horse_scores / deep_analyses / pedigree_analyses / race_level_analyses / pace_analyses`
- 只对上位 `5〜7` 头做结构化推演
- 生成 `race_simulation`
- 不会修改 `horse_scores / marks / strategy`
- LLM 失败时自动 fallback 到 template simulation

`Simulation Review Feedback v1` 会：
- 在赛后基于 `race_simulation` 生成 `simulation_review`
- 检查有利馬 / リスク馬是否兑现
- 检查 `◎○▲` 的 top3 scenario 命中等级
- 生成 simulation 向け的 `new_lessons`
- 不会修改既有 prediction scoring / marks

lesson memory 现在会自动做旧格式 migration，并维护：
- `lesson_id`
- `enabled`
- `used_count`
- `success_count`
- `failure_count`
- `score`

lesson `score` 用于控制长期可用性与检索优先级：
- `analysis` 实际使用 lesson 时会增加 `used_count`
- `review` 后会根据命中情况小幅调整 `score`
- `disabled` 的 lesson 不会参与后续检索
- `lessons-prune` 不会删除 lesson，只会把低分 lesson 设为 `enabled=false`

如果你之前已经生成过错误的 `unknown` lesson，建议先清理对应记录后再重跑：

1. 删除 `source_race_id=202605020811` 且 `course=unknown` 的 lesson，或直接备份后重置 `memory/lessons.json`
2. 重新执行：

```bash
python -m keiba_llm_agent.main review-url --url "https://race.netkeiba.com/race/result.html?race_id=202605020811"
```

## 可选参数

- `analysis --output`: 自定义 prediction 输出路径
- `analysis --lessons`: 自定义 lessons 文件路径
- `review --prediction`: 自定义 prediction 输入路径
- `review --output`: 自定义 review 输出路径
- `review --lessons`: 自定义 lessons 文件路径
- `parse-url --url`: 待解析的 netkeiba URL
- `parse-html --html`: 本地 netkeiba 出马表 HTML 路径
- `parse-html --race-id`: 可选传入 race_id，优先级高于 HTML 内提取结果
- `fetch-race --url`: netkeiba 出马表 URL
- `fetch-race --force-refresh`: 忽略现有 cache，强制重新下载 HTML
- `fetch-race --with-recent-runs`: 抓取每匹马近走并写回 race_data
- `fetch-race --recent-run-limit`: 每匹马抓取近走条数，默认不限制
- `analyze-url --url`: netkeiba 出马表 URL，直接产出 race_data 与 prediction
- `analyze-url --force-refresh`: 忽略现有 cache，强制重新下载 HTML
- `analyze-url --dry-run`: 只保存 race_data，不执行 analysis
- `analyze-url --with-recent-runs`: analysis 前抓取每匹马近走
- `analyze-url --recent-run-limit`: 每匹马抓取近走条数，默认不限制
- `fetch-horse --horse-id`: 单独抓取一匹马的页面
- `fetch-horse --limit`: recent_runs 返回条数，默认不限制
- `fetch-horse --force-refresh`: 忽略现有 horse cache，强制重新下载 HTML
- `parse-pedigree --horse-id`: 单独解析一匹马的父・母・母父
- `parse-pedigree --horse-name`: 可选传入 horse_name
- `parse-pedigree --force-refresh`: 忽略现有 pedigree cache，强制重新下载 pedigree HTML
- `fetch-result --url`: netkeiba result URL
- `fetch-result --force-refresh`: 忽略现有 result cache，强制重新下载 HTML
- `review-url --url`: netkeiba result URL，直接产出 result 与 review
- `review-url --force-refresh`: 忽略现有 result cache，强制重新下载 HTML
- `report-prediction --race-id`: 按 race_id 读取 prediction.json 并生成 Markdown 报告
- `report-prediction --prediction`: 自定义 prediction.json 路径
- `report-prediction --output`: 自定义 Markdown 输出路径
- `report-review --race-id`: 按 race_id 读取 prediction/review 并生成 Markdown 回顾报告
- `report-review --prediction`: 自定义 prediction.json 路径
- `report-review --review`: 自定义 review.json 路径
- `report-review --output`: 自定义 Markdown 输出路径
- `social-prediction --race-id`: 按 race_id 读取 prediction 并生成赛前短文 txt
- `social-prediction --prediction`: 自定义 prediction.json 路径
- `social-prediction --output`: 自定义 txt 输出路径
- `social-review --race-id`: 按 race_id 读取 prediction/review 并生成赛后短文 txt
- `social-review --prediction`: 自定义 prediction.json 路径
- `social-review --review`: 自定义 review.json 路径
- `social-review --output`: 自定义 txt 输出路径
- `predict-race --url`: 一键抓取 shutuba、补 recent_runs、生成 prediction/report/social
- `predict-race --force-refresh`: 忽略现有 cache，强制重新下载 HTML
- `predict-race --recent-run-limit`: 每匹马抓取近走条数，默认不限制
- `predict-race`: 默认不自动生成 report
- `predict-race --enable-report`: 显式生成 prediction report
- `predict-race --skip-report`: 跳过 prediction report 生成
- `predict-race --skip-social`: 跳过 prediction social post 生成
- `predict-race --scoring-profile`: 指定 `accuracy_default / safe_baseline`
- `predict-race --scoring-mode`: 指定 `base_only / pedigree_only / race_level_only / pace_only / current_full / candidate_default / custom`
- `predict-race --pedigree-weight / --race-level-weight / --pace-weight`: 覆盖当前 scoring mode 的默认权重
- `review-race --url`: 一键抓取 result、生成 review/report/social
- `review-race --force-refresh`: 忽略现有 result cache，强制重新下载 HTML
- `review-race`: 默认不自动生成 report
- `review-race --enable-report`: 显式生成 review report
- `review-race --skip-report`: 跳过 review report 生成
- `review-race --skip-social`: 跳过 review social post 生成
- `predict-race --llm-provider`: 指定 `mock`、`gemini` 或 `openai`，覆盖环境变量
- `review-race --llm-provider`: 指定 `mock`、`gemini` 或 `openai`，覆盖环境变量
- `result.json`: 现在会尽量保存 `payouts`、完整 `finish_order` 与 `warnings`
- `review.json`: 当 `bet_hit=true` 但払戻缺失时，会设置 `payout_warning=true` 与 `review_warnings`
- `simulation_review`: 会优先使用 `finish_order` 判断 `top3 / close / mid / failed`，尽量减少 `unknown`
- `validate-race-data --race-id`: 按 race_id 读取 `data/race_data/{race_id}.json` 并输出数据质量检查结果
- `validate-race-data --race-data`: 自定义 race_data.json 路径
- `audit-race-data-flow --race-id`: 审计单场 `race_data / prediction / result / review` 的来源、一致性、评分参与状态
- `audit-data-flow --from-date / --to-date`: 批量审计指定期间的数据闭环完整性
- `llm-check`: 检查当前 provider/model，并发起最小 JSON 测试请求
- `daily-summary --date`: 读取指定日期的 predictions / reviews / results / lessons，并生成日次汇总 markdown 与社交短文
- `daily-summary`: 默认不自动生成 report
- `daily-summary --enable-report`: 显式生成 `data/daily_reports/{date}.md`
- `daily-summary --skip-report`: 跳过 `data/daily_reports/{date}.md` 生成
- `daily-summary --skip-social`: 跳过 `data/social_posts/{date}_daily.txt` 生成
- `daily-summary`: 若部分 race 缺少払戻数据，会提示 `ROI may be unreliable`
- `backtest --from-date / --to-date`: 比较指定期间内 `base_only / pedigree_only / full_adjusted` 三种排序模式
- `backtest --mode`: 仅输出指定 mode，可重复传入
- `backtest --min-races`: 低于该样本数时在 warnings 中提示
- `backtest`: 默认输出 `data/backtests/{from}_{to}.json` 与 `data/backtests/{from}_{to}.md`
- `backtest / backtest-weights`: 若 review 中存在 `payout_warning=true`，ROI 会标记为 `暫定 / unreliable`
- `backtest-weights --from-date / --to-date`: 对 `base_only / current_full / conservative_full / no_pace / no_race_level / race_level_only / pace_only / pedigree_only / candidate_default / candidate_default_recovered` 做权重比较
- `backtest-weights --pedigree-weight / --race-level-weight / --pace-weight`: 追加 `custom` mode，比较自定义权重组合
- `backtest-weights`: 默认输出 `data/backtests/{from}_{to}_weight_tuning.json` 与 `data/backtests/{from}_{to}_weight_tuning.md`
- `missed-top3-analysis --from-date / --to-date`: 分析实际 `result_top3` 中漏出 `predicted topN` 的马
- `missed-top3-analysis --scoring-mode`: 按指定评分模式重排并分析 `base_only / pedigree_only / race_level_only / pace_only / current_full / candidate_default / custom`
- `missed-top3-analysis --pedigree-weight / --race-level-weight / --pace-weight`: custom 或 override 权重
- `missed-top3-analysis --min-popularity`: 只分析 `popularity >= N` 的漏网马
- `missed-top3-analysis --finish`: 只分析指定着顺 `1 / 2 / 3`
- `missed-top3-analysis --top-n`: 默认按 `top5`，也可以改成 `top6 / top7` 看 capture 变化
- `missed-top3-analysis`: 默认输出 `data/error_analysis/{from}_{to}_missed_top3.json` 与 `data/error_analysis/{from}_{to}_missed_top3.md`

### Data Flow Audit

- 目的:
  - 确认每场比赛的 `race_data / prediction / result / review` 是否齐全
  - 确认 `recent_runs` 是否存在赛后数据泄漏
  - 确认 `marks` 是否与 `total_score` 排序一致
  - 确认 `odds / popularity` 是否只作为参考字段，未进入默认 ranking score
  - 确认 `payouts / finish_order` 是否完整，判断 ROI 与 simulation_review 是否可靠
- 单场:
  - `python -m keiba_llm_agent.main audit-race-data-flow --race-id 202605021211`
- 批量:
  - `python -m keiba_llm_agent.main audit-data-flow --from-date 2026-05-01 --to-date 2026-05-31`
- 输出:
  - `data/audits/{race_id}_data_flow_audit.json`
  - `data/audits/{race_id}_data_flow_audit.md`
  - `data/audits/{from}_{to}_data_flow_audit.json`
  - `data/audits/{from}_{to}_data_flow_audit.md`
- 注意:
  - 只读取已有文件，不调用 LLM
  - 不重新抓取 netkeiba
  - 不修改 prediction / result / review

### Scoring Mode

- 当前 v1.0 accuracy default:
  - `scoring_mode = candidate_default`
  - `conditional_weight_profile = candidate_default_v2`
  - base weights: `pedigree_weight = 0.2 / race_level_weight = 1.0 / pace_weight = 0.2`
  - effective weights:
    - 芝: `race_level_weight = 1.2`
    - ダート: `pedigree_weight = 0.1 / pace_weight = 0.8`
    - 14頭以上: `race_level_weight = 1.0`
  - `use_market_score_in_ranking = false`
  - `market_signal_weight = 0.0`
- 理由:
  - Agent LLMの744レース回溯では、中央ダートの展開補正と地方の過剰な展開補正カットが本命命中を改善
  - `race_level` は芝でやや強め、ダートでは標準に抑える
  - `pace` は中央ダートでは反映し、地方では順位ぶれを避けるため外す
  - `pedigree` は中央芝と地方で `0.2`、ダートでは `0.1` に抑える
- `accuracy_default`:
  - `scoring_mode = candidate_default`
  - `borderline_recovery_enabled = false`
  - 当前の accuracy-oriented 推奨設定
- `local_accuracy_default`:
  - `scoring_mode = local_candidate_default`
  - base weights: `pedigree_weight = 0.2 / race_level_weight = 0.6 / pace_weight = 0.0`
  - `borderline_recovery_enabled = false`
  - Agent LLM地方440レース回溯で、旧設定より本命勝率と本命Top3率を優先して改善
- `safe_baseline`:
  - `scoring_mode = base_only`
  - `borderline_recovery_enabled = false`
  - 最保守的安全基线
- `candidate_default_v2` 是当前候选默认配置，不是最终固定结论
- `candidate_default_recovered` は任意の比較用設定
  - 在不改全局权重的前提下，尝试修复 `rank=6 / gap<=1.0` 的保守型漏网
  - 更适合追求 `印内Top3平均 / Top5 winner率` 的比较分析
- `predict-race` / `analysis` / `analyze-url` 都支持 `--scoring-profile` 和 `--scoring-mode`
- 显式指定 `--pedigree-weight / --race-level-weight / --pace-weight` 时，会关闭条件型权重 profile
- `custom` mode 下可用 `--pedigree-weight / --race-level-weight / --pace-weight` 自由覆盖
- 使用例:
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-profile accuracy_default`
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-profile safe_baseline`
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-mode candidate_default`
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-mode base_only`
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-mode custom --pedigree-weight 0.2 --race-level-weight 1.0 --pace-weight 0.0`
- `race_level` 默认保留，`pace` 当前默认仅保留分析层解释，不计入总分
- `odds / popularity` 默认仅作为参考情報、error analysis、ROI / strategy analysis 字段保留，不直接参与核心排序
- `race_level` 补正上限为 `+1.0 / -1.0`
- `pace` 补正上限为 `+0.8 / -0.8`
- `backtest-weights` 可用于比较 `current_full`、保守权重与关闭单模块后的表现
- 这仍然不是正式 ML 模型，权重需要继续通过回顾结果调整
- `lessons-list`: 列出当前 lesson memory，包括 enabled / score / used_count / success_count / failure_count
- `lessons-disable --lesson-id`: 禁用指定 lesson
- `lessons-enable --lesson-id`: 启用指定 lesson
- `lessons-prune --min-score`: 禁用低于给定 score 的 lesson，但不物理删除

### Top5 Borderline Recovery

- 目的：减少 `predicted_rank=6` 且 `score_gap_to_top5<=1.0` 的保守型漏网
- 只处理 `rank=6` 的马，每场最多修复 `1` 匹
- 需要多个正面信号同时成立，避免去追 `rank>=7` 的大穴
- 不修改 `base_total_score`，也不修改 raw adjustment
- 默认只在 `candidate_default / custom` 预测模式中启用
- 社交短文不显示该补正，只体现在 `marks / top5 / report`
- 使用例:
  - `python -m keiba_llm_agent.main predict-race --url "https://race.netkeiba.com/race/shutuba.html?race_id=..." --scoring-mode candidate_default --enable-borderline-recovery`
  - `python -m keiba_llm_agent.main backtest-weights --from-date 2026-05-01 --to-date 2026-05-31 --enable-borderline-recovery`
  - `python -m keiba_llm_agent.main missed-top3-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default --simulate-borderline-recovery`

### Missed Top3 Error Analysis

- 用于分析实际 `result_top3` 中哪些马没有进入 `predicted topN`
- 目标是找出把 `印内Top3平均` 从当前水平继续往 `2.0` 推进的方向
- 只读取已有 `prediction / result / review`，不修改原文件
- 不调用 LLM
- 不重新抓取 netkeiba
- 使用例:
  - `python -m keiba_llm_agent.main missed-top3-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default`
  - `python -m keiba_llm_agent.main missed-top3-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode custom --pedigree-weight 0.2 --race-level-weight 1.0 --pace-weight 0.0`
  - `python -m keiba_llm_agent.main missed-top3-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default --top-n 6`
  - `python -m keiba_llm_agent.main missed-top3-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default --simulate-borderline-recovery`
- 输出:
  - `data/error_analysis/{from}_{to}_missed_top3.json`
  - `data/error_analysis/{from}_{to}_missed_top3.md`

### Deep Miss Analysis

- 用于分析实际 `Top3` 但预测 `rank>=7` 的低排位漏马
- 重点区分：
  - `rank 7-8`：轻度漏马
  - `rank 9-12`：中度漏马
  - `rank 13+`：深度漏马
- 目标是找出哪些真实好走马一开始就被排得太低，以及共同特征是什么
- 只读取已有 `prediction / result / review`，不修改原文件
- 不调用 LLM
- 不重新抓取 netkeiba
- 使用例:
  - `python -m keiba_llm_agent.main deep-miss-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default`
  - `python -m keiba_llm_agent.main deep-miss-analysis --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode custom --pedigree-weight 0.2 --race-level-weight 1.0 --pace-weight 0.0`
- 输出:
  - `data/error_analysis/{from}_{to}_deep_miss_top3.json`
  - `data/error_analysis/{from}_{to}_deep_miss_top3.md`

### Deep Miss Rule Simulation

- 用于验证 `rank>=7` 的低排位漏马是否存在可控的 safety net 规则
- 不会修改正式预测
- 不会影响 `predict-race` 默认输出
- 不调用 LLM
- 不接入新数据源
- 使用例:
  - `python -m keiba_llm_agent.main deep-miss-rule-simulate --from-date 2026-05-01 --to-date 2026-05-31 --baseline-mode candidate_default_recovered`
  - `python -m keiba_llm_agent.main deep-miss-rule-simulate --from-date 2026-05-01 --to-date 2026-05-31 --scoring-mode candidate_default`
  - `python -m keiba_llm_agent.main deep-miss-rule-simulate --from-date 2026-05-01 --to-date 2026-05-31 --baseline-mode custom --pedigree-weight 0.2 --race-level-weight 1.0 --pace-weight 0.0`
- 输出:
  - `data/error_analysis/{from}_{to}_deep_miss_rule_simulation.json`
  - `data/error_analysis/{from}_{to}_deep_miss_rule_simulation.md`
- 目的:
  - 比较多种 low-rank safety net 规则的 what-if 效果
  - 只在 improvement 明显、worse 少、replaced_top3 少时，才考虑下一阶段是否进入正式规则
  - `odds_undervalued_limited` 仅作为 watchlist 用规则，不建议直接进入 default ranking

### Penalty Refinement Simulation

- 用于验证 `HEAD_TO_HEAD_NEGATIVE`、`DISTANCE_UNKNOWN`、`TRACK_CONDITION_UNKNOWN` 是否在强正面信号马身上惩罚过重
- 只做 what-if simulation，不修改正式预测、不修改 scoring 默认值
- 不调用 LLM，不重新抓取 netkeiba
- 默认 baseline 使用 `candidate_default_recovered`
- 比较规则:
  - `head_to_head_negative_cap`: 强正面信号时把 `HEAD_TO_HEAD_NEGATIVE` 负补正最多缓和到 `-0.2`
  - `head_to_head_zero_floor`: 更激进地缓和到 `0.0`，仅用于对照
  - `distance_unknown_stamina_exception`: 长距离且血统スタミナ/近走/コース支持时，模拟减轻 `DISTANCE_UNKNOWN`
  - `track_condition_unknown_soften`: 强正面信号时模拟减轻 `TRACK_CONDITION_UNKNOWN`
  - `positive_stack_protection`: 多模块正面信号很厚时，验证是否被 risk 压得过低
  - `combined_penalty_refinement`: 合成上述规则的上限模拟
- 使用例:
  - `python -m keiba_llm_agent.main penalty-refinement-simulate --from-date 2026-05-01 --to-date 2026-05-31`
  - `python -m keiba_llm_agent.main penalty-refinement-simulate --from-date 2026-05-01 --to-date 2026-05-31 --baseline-mode candidate_default_recovered`
  - `python -m keiba_llm_agent.main penalty-refinement-simulate --from-date 2026-05-01 --to-date 2026-05-31 --max-rank 10`
- 输出:
  - `data/error_analysis/{from}_{to}_penalty_refinement_simulation.json`
  - `data/error_analysis/{from}_{to}_penalty_refinement_simulation.md`

### Score Recalibration Simulation

- 用于验证“低分但本该高分”的马是否能通过更合理的评分校准进入 Top5
- 只做 what-if simulation，不修改正式预测、不修改 scoring 默认值
- 不调用 LLM，不重新抓取 netkeiba
- 默认 baseline 使用 `candidate_default_recovered`
- 比较规则:
  - `unknown_softening`: 正面信号足够时，把 `DISTANCE_UNKNOWN / COURSE_UNKNOWN / TRACK_CONDITION_UNKNOWN` 视为未知而非强负面
  - `positive_stack_boost`: 近走、条件、血统、race_level、pace 等多模块正面信号叠加时小幅加分
  - `risk_cap_positive_stack`: 正面信号厚的马，验证是否被 risk penalty 过度压低
  - `conditional_pace_support`: pace 只在近走或 race_level 也支持时小幅进入模拟分
  - `race_level_recent_synergy`: race_level positive 与近走稳定同时成立时小幅加分
  - `combined_recalibration`: 合成上述规则，单马补正上限 `1.5`
- 使用例:
  - `python -m keiba_llm_agent.main score-recalibration-simulate --from-date 2026-05-01 --to-date 2026-05-31`
  - `python -m keiba_llm_agent.main score-recalibration-simulate --from-date 2026-05-01 --to-date 2026-05-31 --baseline-mode candidate_default_recovered`
  - `python -m keiba_llm_agent.main score-recalibration-simulate --from-date 2026-05-01 --to-date 2026-05-31 --baseline-mode custom --pedigree-weight 0.2 --race-level-weight 1.0 --pace-weight 0.0`
- 输出:
  - `data/error_analysis/{from}_{to}_score_recalibration_simulation.json`
  - `data/error_analysis/{from}_{to}_score_recalibration_simulation.md`
- 判断标准:
  - `simulated avg` 是否高于 baseline
  - `better_race_count > worse_race_count`
  - `replaced_top3_count` 是否可控
  - 只有多期间稳定有效时，才考虑进入正式 scoring

## 测试

```bash
python -m unittest tests/test_keiba_llm_agent.py
```
