# Keiba Pipeline

End-to-end local workflow for JRA race data: build race cards, collect history, generate predictions, fetch odds, produce a betting plan, and log results.

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
`race_card.py` -> `new_history.py` -> `predictor.py` -> `odds_extract.py` -> `pipeline/bet_plan_update.py`

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

## Setup
Install dependencies:
```
python -m pip install pandas numpy scikit-learn lightgbm beautifulsoup4 selenium fastapi uvicorn
```

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
- Writes bet stats: `bet_ticket_results.csv`, `bet_type_stats.csv`, `race_results.csv`, `wide_box_results.csv`
- Runs both optimizers and `offline_eval`

### Optimize and evaluate
```
python pipeline/optimize_params.py
python pipeline/optimize_predictor_params.py
python pipeline/offline_eval.py
```

## Configuration
- Bet plan config: `pipeline/config_<scope>.json`
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
| `BET_STRATEGY` | Override bet strategy | `BET_STRATEGY=balanced` |
| `PREDICTOR_STRATEGY` | Override predictor strategy | `PREDICTOR_STRATEGY=steady` |
| `PIPELINE_SHARED_CHROME` | Disable shared Chrome (`0` = off) | `PIPELINE_SHARED_CHROME=0` |
| `PIPELINE_HEADLESS` | Headless mode toggle | `PIPELINE_HEADLESS=1` |
| `PIPELINE_CHROME_PROFILE` | Reuse Chrome profile dir | `PIPELINE_CHROME_PROFILE=D:\\keiba\\chrome_profile` |
| `PIPELINE_SKIP_COOKIE_INJECTION` | Inject cookies when set to `0` | `PIPELINE_SKIP_COOKIE_INJECTION=0` |
| `PIPELINE_SCRAPE_DELAY` | Fixed delay between scraping steps | `PIPELINE_SCRAPE_DELAY=3` |
| `CHROME_BIN` / `GOOGLE_CHROME_BIN` / `CHROME_PATH` | Chrome binary override | `CHROME_BIN=C:\\Path\\To\\chrome.exe` |
| `CHROME_DEBUGGER_ADDRESS` | Attach to existing Chrome | `CHROME_DEBUGGER_ADDRESS=127.0.0.1:9222` |
| `BET_OPT_WINDOW` / `BET_OPT_MIN_SAMPLES` / `BET_OPT_MAX_ROI_STD` | Bet optimizer tuning | `BET_OPT_WINDOW=10` |
| `PRED_OPT_WINDOW` / `PRED_OPT_MIN_SAMPLES` / `PRED_OPT_MAX_STD` | Predictor optimizer tuning | `PRED_OPT_WINDOW=10` |
| `PRED_OPT_RISK_SLOW` / `PRED_OPT_RISK_FREEZE` | Predictor risk gates | `PRED_OPT_RISK_FREEZE=0.2` |

## Web UI
```
python pipeline/web_server.py
```
- Requires `fastapi` and `uvicorn`
- Runs on `http://127.0.0.1:8000`

## Data and Outputs
- Root outputs (latest run): `predictions.csv`, `odds.csv`, `fuku_odds.csv`, `wide_odds.csv`, `quinella_odds.csv`, `trifecta_odds.csv`
- Pipeline outputs: `pipeline/bet_plan_update.csv`
- Per-scope logs: `pipeline/data/<scope>/runs.csv`, `results.csv`, `predictor_results.csv`, and optimizer logs
- Per-race snapshots: `pipeline/data/<scope>/<race_id>/` (predictions, odds, bet plans)

## Notes
- `odds_extract.py` is Selenium-based.
- `cookie.txt` is a JSON array of cookies with `name` and `value` fields:
  ```json
  [
    {"name": "example", "value": "..."}
  ]
  ```
- Root CSV outputs are overwritten each run; per-race snapshots are kept under `pipeline/data/<scope>/`.
