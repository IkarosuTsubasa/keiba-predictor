本地更新流程（无网?）

Browser/automation options (pipeline, Selenium)

- Shared Chrome (default ON): uses one Chrome instance per pipeline.

  - Disable: set PIPELINE_SHARED_CHROME=0

- Headless mode: default is OFF (visible Chrome window).

  - Enable: set PIPELINE_HEADLESS=1

- Fixed Chrome profile (reuse login):

  - Set PIPELINE_CHROME_PROFILE=D:\keiba\chrome_profile

  - Do not open regular Chrome with the same profile directory.

  - When set, cookie.txt injection is skipped automatically.


- Cookie injection default is OFF.

  - PIPELINE_SKIP_COOKIE_INJECTION=0 to force cookie.txt injection

  - PIPELINE_SKIP_COOKIE_INJECTION=1 to skip cookie.txt

- Delay between scraping steps:

  - PIPELINE_SCRAPE_DELAY=3  (seconds; overrides random delay)

- Chrome path override:

  - CHROME_BIN=C:\Path\To\chrome.exe

- 403/429 handling:

  - Selenium scripts stop immediately if a block page is detected.

- odds_extract.py is Selenium-only (no urllib fallback).

- record_pipeline.py does not auto-start shared Chrome.

  - To attach manually: set CHROME_DEBUGGER_ADDRESS=127.0.0.1:<port>





1) ?行主流程

   python update/run_pipeline.py

   - ?行?序：出?表.py -> new?史?.py -> ??新.py -> odds_extract.py -> update/bet_plan_update.py

   - ?出：update/bet_plan_update.csv

   - 日志：update/data/runs.csv

   - ??策略自?从 update/config_{surface}.json ??

   - ??策略自?从 update/predictor_config_{surface}.json ??

   - odds.csv 会自?快照保存到 update/data/odds_{run_id}.csv



2) 比??束后??收益

   python update/record_result.py

   - ?入 run_id（默?最新）

   - ?入盈利/??（日元）

   - 保存到：update/data/results.csv

   - 会自??行??参数?化



3) 比??束后????准?度（推荐）

   python update/record_predictor_result.py

   - ?入真? Top3 ?名（按 1/2/3 名?序）

   - 保存到：update/data/predictor_results.csv

   - 会自??行??参数?化



2+3 合并??（推荐）

   python update/record_pipeline.py

   - 一次性?入收益 + 真? Top3

   - 自??行??参数?化 + ??参数?化

   - 自??行???估（默?窗口 N=10）

   - 会?????法命中情况并写入??

   - 盈?可留空，若 odds.csv 存在?会用?率估算（?近似）



4) ?化??方案参数

   python update/optimize_params.py

   - 更新：update/config_{surface}.json

   - 日志：update/data/config_history.csv



5) ?化??参数

   python update/optimize_predictor_params.py

   - 更新：update/predictor_config_{surface}.json

   - 日志：update/data/predictor_config_history.csv
   - Env: PRED_OPT_WINDOW / PRED_OPT_MIN_SAMPLES / PRED_OPT_MAX_STD
   - Optimizer quality: 0.55*rank_score + 0.35*ev_score + 0.10*hit_rate



6) ???估（最近 N ??定性）

   python update/offline_eval.py

   - 默?窗口 N=10，可自行?入

   - ?出：update/data/offline_eval.csv



?明

- 自?更新只?整参数，不会自?改写算法??。

- ??方案算法在 update/bet_plan_update.py。

- ??算法在 ??新.py。

- ?化脚本默?使用最近 10 ?，并要求?本数?5且波?不?大才会?整。
- Optimizer now uses soft gate on volatility/drawdown (step shrink) and always writes history.

- ??方案?化会参考各?法命中率，???整?重。

- ?法??日志：update/data/bet_ticket_results.csv、update/data/bet_type_stats.csv、update/data/race_results.csv。

- ???分??合指?：rank_score + ev_score + hit_rate（?重 0.4/0.4/0.2），并采用 EMA 作??化基准。
- Optimizer quality uses weights: 0.55/0.35/0.10 (rank/ev/hit) for EMA/STD/DD.

- ??自信度 confidence_score 也会??到 predictor_results.csv（包含 stability/validity/consistency + rank_ema/ev_ema/risk_score）。

- ?算：confidence = sqrt(stability * validity) * consistency；stability=risk_score，validity=0.6*rank_ema+0.4*ev_ema，consistency=Top1-Top3 概率差/0.15（clamp 到 0-1）。

- ?参器会分?看 rank_ema / ev_ema / risk_score 来决定“?定 vs 价? vs ?控”方向。

- ?控机制：risk_score < 0.4 会??本次更新，并保存?参原因到 predictor_config_history.csv。

- ?次更新前会??旧配置到 update/predictor_config_prev_{surface}.json，便于回?。

- ev_score 由 odds.csv 的?率估算，属于近似指?（非官方配当）。

- 固定??策略：update/config_{surface}.json 里?置 selector.epsilon = 0 和 active_strategy。

- 固定??策略：update/predictor_config_{surface}.json 里?置 selector.epsilon = 0 和 active_strategy。

- 手???可使用?境?量：BET_STRATEGY / PREDICTOR_STRATEGY。

初始化

1) 首次初始化

   python update/init_update.py

   - ?建 update/data 及各?日志CSV（若不存在）

   - 若配置文件不存在会写入默?配置



2) 重置日志（可?）

   python update/init_update.py --reset

   - 清空并重建日志CSV



?出字段?明（重点）



predictions.csv（???出）

- HorseName：?名

- agg_score：?史表?聚合分（recent_race_count ?内? top_score_count 的 record_score 均?）

- best_TimeIndexEff：?本内最高 TimeIndexEff

- avg_TimeIndexEff：?本内 TimeIndexEff 平均

- best_Uphill：?本内最小 Uphill（越小越好）

- dist_close：与本?距?差的平均?（越小越?近）

- races_used：参与聚合的?本数

- Top3Prob_model: model Top3 probability (primary)
- Top3Prob_est: legacy field (optional)

- confidence_score：本???自信度（全局指?，所有行相同）

- stability_score：?定性 = risk_score

- validity_score：0.6*rank_ema + 0.4*ev_ema

- consistency_score：Top1-Top3 概率差 / 0.15（clamp 到 0-1）

- rank_ema / ev_ema / risk_score：?参器??的最新状??



update/data/predictor_results.csv（?后?价）

- top3_hit_count：Top3 命中数量

- top1_hit：??第 1 名是否命中

- top1_in_top3：??第 1 名是否?入前三

- top3_exact：?? Top3 是否完全一致（忽略?序）

- rank_score：NDCG@3 排名?量（越高越好）

- ev_score：?率价??分（tanh ?一化）

- hit_rate：Top3 命中率（命中数 / 3）

- score_total：?合分（0.4*rank + 0.4*ev + 0.2*hit）

- confidence_score / stability_score / validity_score / consistency_score：同上（?????的自信度）



update/data/bet_ticket_results.csv / bet_type_stats.csv（下注命中）

- hit：是否命中

- est_payout_yen：估算回收金?（基于 odds.csv 的近似?）

- est_profit_yen：估算盈?（回收 - 投入）



Surface separation

- Logs: update/data/dirt/ or update/data/turf/

- Configs: update/config_dirt.json, update/config_turf.json

- Predictor configs: update/predictor_config_dirt.json, update/predictor_config_turf.json

- Use SURFACE_KEY=dirt|turf or input prompt when running scripts.


Location separation
- Logs: update/data/{surface}/{location}/
- Configs: update/config_{surface}_{location}.json
- Predictor configs: update/predictor_config_{surface}_{location}.json
- Use LOCATION_KEY to select location (default=default).

