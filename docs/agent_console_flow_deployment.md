# Console 到 AI 予測全流程部署清单

## 目标流程

1. 管理员在 `/keiba/console` 的 `レースID一括登録` 登记 race_id。
2. Render cron 每 5 分钟调用一次 `run_due`。
3. cron 自动补全 race meta，并在发走前 1 小时把任务派发到 GitHub Actions。
4. GitHub Actions 运行 `keiba_llm_agent`，生成 `data/predictions/{race_id}.json` 后 callback 回 Render。
5. Render 保存预测，任务进入 `AI予測完了`。
6. 发走后 15 分钟，Render 直接抓取结果并保存 `data/results/{race_id}.json`。
7. public 页面读取预测与结果，`履歴分析` 统计网站预测命中率。

## 已有 Render 服务

当前 `render.yaml` 已包含需要的两个服务：

- `keiba-web`: Web 服务，build 命令为 `pip install -r requirements.txt && cd frontend && npm install --include=dev && npm run build`，启动命令为 `cd pipeline && python web_server.py`。
- `keiba-run-due-cron`: 唯一 cron 服务，计划为 `*/5 * * * *`，启动命令为 `cd pipeline && python cron_trigger_run_due.py`。
- persistent disk: `pipeline-data`，挂载到 `/opt/render/project/src/pipeline/data`。

本次流程不需要新增第二个 cron。结果取得、旧任务 cleanup、预测 dispatch 都走同一个 `run_due`。

## Render 环境变量

这些变量继续沿用既有远程任务配置：

- `ADMIN_TOKEN`: cron 调用 `/internal/run_due` 的鉴权 token。
- `PIPELINE_PUBLIC_SITE_URL`: 公开站点根地址，例如 `https://www.ikaimo-ai.com`。
- `PIPELINE_CALLBACK_SECRET`: GitHub Actions callback HMAC secret。
- `GITHUB_ACTIONS_OWNER`: GitHub owner。
- `GITHUB_ACTIONS_REPO`: GitHub repo。
- `GITHUB_ACTIONS_TOKEN`: 允许 dispatch workflow 的 GitHub token。
- `RUN_DUE_TRIGGER_URL`: cron 服务使用，`https://<your-domain>/internal/run_due`。Web 服务不需要依赖它。
- `GITHUB_ACTIONS_REF`: 可选，默认 `main`。
- `GITHUB_ACTIONS_AGENT_WORKFLOW`: 可选，默认 `agent-prediction-remote.yml`。
- `KEIBA_AGENT_PREDICTIONS_DIR`: 可选，默认写入 `pipeline/data/agent_predictions`。
- `KEIBA_AGENT_RESULTS_DIR`: 可选，默认写入 `pipeline/data/agent_results`。
- `PIPELINE_AUTO_PREDICTION_NOTIFY_ENABLED`: 可选，默认关闭。只有设为 `1/true/yes/on` 时，预测完成后才会自动判断是否发送通知。
- `PIPELINE_AUTO_PREDICTION_NOTIFY_MIN_CONFIDENCE`: 可选，默认 `0.62`。只有达到 public 页面 `高評価` 分界的预测才会发送 ntfy/FCM 通知。agent 预测中 `BET` 会直接视为高評価，`SKIP` 不发送。

如果线上已经配置过 v5 remote predictor，通常只需要确认旧变量仍然存在。预测和结果默认会写入 Render persistent disk，所以不额外配置 `KEIBA_AGENT_PREDICTIONS_DIR` 与 `KEIBA_AGENT_RESULTS_DIR` 也可以。

## GitHub Actions 配置

需要保留并启用 workflow：

- `.github/workflows/agent-prediction-remote.yml`

GitHub Secrets 至少需要：

- `GEMINI_API_KEY`: `keiba_llm_agent` 调用 Gemini 生成公开预测文案时使用。
- `PIPELINE_CALLBACK_SECRET`: 必须与 Render 的 `PIPELINE_CALLBACK_SECRET` 一致。

GitHub workflow dispatch input 由 Render 自动传入：

- `task_id`
- `race_id`
- `race_url`
- `callback_url`

## Console 运用

1. 打开 `/keiba/console`。
2. 在 `レースID一括登録` 中粘贴 race_id 或 netkeiba URL。
3. 登记后任务先进入情報補完待ち或最終予想待ち。
4. cron 到点后自动进入 `AI予測` dispatch。
5. GitHub Actions callback 成功后显示 `AI予測完了`。
6. 发走后 15 分钟，cron 自动进入结果取得并最终变成 `完了`。
7. 若失败，可在 Console 使用 `AI予測を再実行` 或 `結果取得を再試行`。
8. `AI予測ヘルスチェック` 会显示保存文件数、GitHub task、注册比赛状态、`run_due` 最近执行履歴；也可以手动执行 `期限到来を確認` 与 `run_dueを実行`。

## 数据落点

线上持久化目录：

- race jobs: `/opt/render/project/src/pipeline/data/_shared/race_jobs.json`
- remote tasks: `/opt/render/project/src/pipeline/data/_shared/v5_remote_tasks.json`
- run_due history: `/opt/render/project/src/pipeline/data/_shared/run_due_history.jsonl`
- agent predictions: `/opt/render/project/src/pipeline/data/agent_predictions/{race_id}.json`
- agent results: `/opt/render/project/src/pipeline/data/agent_results/{race_id}.json`

public 页面读取时会优先读取 persistent disk 的生成文件，同时继续兼容旧目录 `data/predictions`、`data/results`、`keiba_llm_agent/data/predictions`、`keiba_llm_agent/data/results`。

## 本地验证

上线前运行：

```bash
python pipeline/smoke_agent_console_flow.py
python -m py_compile pipeline/smoke_agent_console_flow.py pipeline/race_job_store.py pipeline/web_admin/task_routes.py pipeline/web_app.py
```

如果前端显示也有改动，再运行：

```bash
cd frontend && npm run build
```

## 首次数据迁移

如果已有 `keiba_llm_agent/data/predictions` 或 `keiba_llm_agent/data/results` 的旧 JSON，需要先迁移到 persistent disk 默认目录。

本地 dry-run：

```bash
python pipeline/migrate_agent_prediction_data.py --dry-run
```

线上 dry-run：

```bash
ssh -o BatchMode=yes -o UpdateHostKeys=no srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com "cd /opt/render/project/src && python pipeline/migrate_agent_prediction_data.py --dry-run"
```

确认数量后执行迁移：

```bash
ssh -o BatchMode=yes -o UpdateHostKeys=no srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com "cd /opt/render/project/src && python pipeline/migrate_agent_prediction_data.py"
```

## 常见故障

- GitHub dispatch 失败：检查 `GITHUB_ACTIONS_OWNER`、`GITHUB_ACTIONS_REPO`、`GITHUB_ACTIONS_TOKEN`、workflow 文件名。
- callback 403：确认 Render 与 GitHub Secrets 的 `PIPELINE_CALLBACK_SECRET` 完全一致。
- GitHub workflow 成功但无 prediction：检查 `keiba_llm_agent/data/predictions/{race_id}.json` 是否生成。
- 结果未取得：netkeiba 结果页可能还没发布，等待下一次 cron 或手动点击 `結果取得を再試行`。
- public 页面没有新预测：确认 `data/predictions/{race_id}.json` race_id 与页面 race_id 一致。
