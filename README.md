# Keiba Pipeline

End-to-end local workflow for JRA race data: build race cards, collect history, generate predictions, fetch odds, and log results.

## Table of Contents
- Overview
- Project Layout
- Requirements
- Setup
- Usage
- Configuration
- Environment Variables
- Web UI
- Data and Outputs
- Notes

## Overview
Pipeline order (run by `pipeline/run_pipeline.py`):
`race_card.py` -> `new_history.py` -> `predictor.py` -> `odds_extract.py`

## Project Layout
- Root scripts: `race_card.py`, `new_history.py`, `predictor.py`, `odds_extract.py`
- `pipeline/`: orchestration scripts, configs, logs, web UI
- `pipeline/data/<scope>/`: per-scope logs and snapshots
- Root CSVs: inputs like `kachiuma.csv` and `shutuba.csv`; outputs like `predictions.csv` and odds CSVs

## Requirements
- Python 3.x
- Google Chrome (for Selenium)
- Python packages:
  - `pandas`, `numpy`, `scikit-learn`, `lightgbm`
  - `beautifulsoup4`, `selenium`
  - `fastapi`, `uvicorn` (web UI only)
  - `pydantic`, `google-genai` (Gemini agent / policy layer)

## Setup
Install dependencies:
```
python -m pip install pandas numpy scikit-learn lightgbm beautifulsoup4 selenium fastapi uvicorn pydantic google-genai
```

### Gemini Agent / Policy Layer（免费 API）
1. 设置 API Key（Windows PowerShell）：
   `setx GEMINI_API_KEY "your_key_here"`（新终端生效）
2. 可选模型：
   `setx GEMINI_MODEL "gemini-3.1-flash-lite"`（默认就是这个模型）
3. 可选离线/CI 模式：
   `set GEMINI_POLICY_MOCK=1`（仅用本地 deterministic policy）
4. 运行带 Gemini policy 的下注计划：
   Legacy manual-buy command removed from this repository.
5. 默认缓存目录：
   `pipeline/data/policy_cache_gemini/`
6. 运行 smoke：
   `python pipeline/smoke_gemini_policy.py`

## Usage
### Initialize
Creates logs and default configs if missing.
```
python pipeline/init_update.py
python pipeline/init_update.py --reset
```

### Run the full pipeline
```
python pipeline/run_pipeline.py
```

### Keiba LLM Agent condition weight simulation
What-if analysis only. This command compares conditional scoring scenarios without changing prediction, review, or result files. It does not call LLMs and does not scrape netkeiba.
```
python -m keiba_llm_agent.main condition-weight-simulate --from-date 2026-05-01 --to-date 2026-05-31
```
- JSON: `keiba_llm_agent/data/error_analysis/{from}_{to}_condition_weight_simulation.json`
- Markdown: `keiba_llm_agent/data/error_analysis/{from}_{to}_condition_weight_simulation.md`
- Main purpose: verify whether conditional weights, such as turf race-level support or stricter rank6 recovery, are stable enough for later scoring changes.

### Agent LLM Render backtest audit
Use this flow when optimizing Agent LLM scoring, coefficients, confidence, or central/local prediction quality. It keeps historical Render data intact and separates full scoring-logic backtests from saved-score reweighting.

Fresh Render snapshots should be downloaded into a new directory under `analysis_outputs/`, for example:
```
powershell -ExecutionPolicy Bypass -File scripts\download_render_agent_data.ps1 -WorkspaceRoot D:\keiba\new
```

Rebuild audit bundles for a downloaded snapshot:
```
python pipeline\build_agent_backtest_audit.py ^
  --predictions-dir analysis_outputs\<snapshot>\agent_predictions ^
  --results-dir analysis_outputs\<snapshot>\agent_results ^
  --audit-dir analysis_outputs\<snapshot>\agent_backtest_audit ^
  --race-data-dir keiba_llm_agent\data\race_data
```

Audit interpretation:
- Audit inputs are grouped by `race_id`; `final` is preferred over `early`, and only one effective sample per race is counted. Files below `_versions` are preserved evidence, not additional samples.
- `audit_kind=full`: raw `race_data` exists, so scoring logic can be recomputed.
- `audit_kind=prediction_only`: only saved prediction scores exist, so use it for coefficient/weight replay only.
- Always report sample counts, settled result coverage, central/local split, and missing inputs before recommending changes.
- Do not delete or overwrite `pipeline/data/agent_predictions`, `pipeline/data/agent_results`, `pipeline/data/agent_backtest_audit`, `keiba_llm_agent/data/backtest_audit`, or `analysis_outputs/` snapshots during optimization.
- Scoring-model optimization uses every settled `full` audit through grouped or repeated cross-validation. A recent-week slice is diagnostic only and does not independently select or reject a model.
- The primary reranker KPI is predicted Top5 coverage of the actual Top3: covered podium slots/recall, all three covered, and at least two covered. Winner-rank metrics are secondary guardrails.
- Odds and popularity are excluded from model features and candidate selection.

Personal Codex skill shortcut:
```
Use $keiba-agent-optimizer to download Render Agent data and run a safe scoring backtest review.
```

The current place-coverage rerankers are trained with:
- `central_turf`: every settled full race, actual Top3 target, equal podium weights, preserve Top1, blend `0.5`.
- `local`: every settled full race, actual Top3 target, winner weight `2.0`, preserve Top2, blend `1.0`.
- `central_dirt`: no reranker is enabled until all-full repeated validation becomes stable.

### Record race results
```
python pipeline/record_result.py
```
- Writes `pipeline/data/<scope>/results.csv`
- Auto-runs `pipeline/optimize_params.py`

### Record prediction accuracy (recommended)
```
python pipeline/record_predictor_result.py
```
- Writes `pipeline/data/<scope>/predictor_results.csv`
- Auto-runs `pipeline/optimize_predictor_params.py`

### Combined recording
```
python pipeline/record_pipeline.py
```
- Records results and prediction accuracy
- Writes race stats: `race_results.csv`, `wide_box_results.csv`
- Runs both optimizers and `offline_eval`

### Optimize and evaluate
```
python pipeline/optimize_params.py
python pipeline/optimize_predictor_params.py
python pipeline/offline_eval.py
```

### Periodic cleanup
Conservative cleanup for runtime caches and non-web-facing experimental artifacts.
```
cd pipeline
python cleanup_runtime_artifacts.py --dry-run
python cleanup_runtime_artifacts.py
```
- Keeps web-facing run history and race snapshots
- Removes superseded early Agent prediction/audit files only after matching final prediction and final audit files both validate successfully
- Add `--skip-agent-early-prediction-cleanup` to keep early Agent versions during a manual cleanup run
- Archives settled/failed/deleted jobs older than 3 days out of the active queue and cleans their stale artifacts, caches, and old experimental files
- Add `--include-offline-research` to also remove old `offline_eval.csv`, `context_dataset.csv`, `context_summary.csv`, and `history_races.csv`
- Add `--include-legacy-debug` to also remove old `bet_engine_v*_cfg_*.json`
- `cleanup_periodic_storage.py` remains as a compatibility alias

## Configuration
- Runtime config: `pipeline/config_<scope>.json`
- Predictor config: `pipeline/predictor_config_<scope>.json`
- No-bet log: `pipeline/no_bet_log_<scope>.csv`
- Optimizer history:
  - `pipeline/data/<scope>/config_history.csv`
  - `pipeline/data/<scope>/predictor_config_history.csv`
  - `pipeline/predictor_config_prev_<scope>.json` (backup)

## Environment Variables
| Name | Purpose | Example |
| --- | --- | --- |
| `SCOPE_KEY` / `SURFACE_KEY` | Select scope (`central_dirt`, `central_turf`, `local`) | `SCOPE_KEY=central_dirt` |
| `PREDICTOR_STRATEGY` | Override predictor strategy | `PREDICTOR_STRATEGY=steady` |
| `PIPELINE_SHARED_CHROME` | Disable shared Chrome (`0` = off) | `PIPELINE_SHARED_CHROME=0` |
| `PIPELINE_HEADLESS` | Headless mode toggle | `PIPELINE_HEADLESS=1` |
| `PIPELINE_CHROME_PROFILE` | Reuse Chrome profile dir | `PIPELINE_CHROME_PROFILE=D:\\keiba\\chrome_profile` |
| `PIPELINE_SCRAPE_DELAY` | Fixed delay between scraping steps | `PIPELINE_SCRAPE_DELAY=3` |
| `CHROME_BIN` / `GOOGLE_CHROME_BIN` / `CHROME_PATH` | Chrome binary override | `CHROME_BIN=C:\\Path\\To\\chrome.exe` |
| `CHROME_DEBUGGER_ADDRESS` | Attach to existing Chrome | `CHROME_DEBUGGER_ADDRESS=127.0.0.1:9222` |
| `PRED_OPT_WINDOW` / `PRED_OPT_MIN_SAMPLES` / `PRED_OPT_MAX_STD` | Predictor optimizer tuning | `PRED_OPT_WINDOW=10` |
| `PRED_OPT_RISK_SLOW` / `PRED_OPT_RISK_FREEZE` | Predictor risk gates | `PRED_OPT_RISK_FREEZE=0.2` |

## Web UI
```
python pipeline/web_server.py
```
- Requires `fastapi` and `uvicorn`
- Runs on `http://127.0.0.1:8000`

## Android Shell Preview
- Android preview project lives in `android/`
- Current approach is a native `Kotlin` shell with a `WebView`
- First-pass tabs:
  - `レース`
  - `履歴分析`
  - `私の日報`
  - `その他`
- Default web target is `https://www.ikaimo-ai.com/keiba`
- See `android/README.md` for details

## Data and Outputs
- Root outputs (latest run): `predictions.csv`, `odds.csv`, `fuku_odds.csv`, `wide_odds.csv`, `quinella_odds.csv`, `trio_odds.csv`
- Per-scope logs: `pipeline/data/<scope>/runs.csv`, `results.csv`, `predictor_results.csv`, and optimizer logs
- Per-race snapshots: `pipeline/data/<scope>/<race_id>/` (predictions and odds)

## Notes
- `odds_extract.py` is Selenium-based.
- Root CSV outputs are overwritten each run; per-race snapshots are kept under `pipeline/data/<scope>/`.
