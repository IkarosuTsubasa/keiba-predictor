import React, { useEffect, useMemo, useState } from "react";
import AdminLoginPage, { ADMIN_TOKEN_STORAGE_KEY } from "./AdminLoginPage";
import PageSectionHeader from "./PageSectionHeader";

function updateWorkspaceUrl(appBasePath, scopeKey, runId) {
  const params = new URLSearchParams();
  if (scopeKey) params.set("scope_key", scopeKey);
  if (runId) params.set("run_id", runId);
  const nextUrl = `${appBasePath}/console/workspace${params.toString() ? `?${params.toString()}` : ""}`;
  window.history.pushState({}, "", nextUrl);
}

function DataTable({ title, rows, maxRows = 8 }) {
  if (!rows?.length) return null;
  const columns = Object.keys(rows[0] || {}).slice(0, 6);
  return (
    <section className="admin-workspace-subtable">
      <h4>{title}</h4>
      <div className="admin-workspace-table">
        <table>
          <thead>
            <tr>
              {columns.map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, maxRows).map((row, index) => (
              <tr key={index}>
                {columns.map((key) => (
                  <td key={key}>{String(row?.[key] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetaRow({ items }) {
  if (!items?.length) return null;
  return (
    <div className="admin-workspace-meta">
      {items.map((item, index) => (
        <span key={`${item}-${index}`}>{item}</span>
      ))}
    </div>
  );
}

function notifyMeta(data) {
  const status = String(data?.ntfy_notify_status || "").trim().toLowerCase();
  if (status === "notified") {
    return {
      label: "送信済み",
      tone: "good",
      detail: data?.ntfy_notify_engine ? `エンジン: ${data.ntfy_notify_engine}` : data?.ntfy_notified_at || "",
    };
  }
  if (status === "failed") {
    return {
      label: "送信失敗",
      tone: "danger",
      detail: data?.ntfy_notify_error || "",
    };
  }
  return {
    label: "未送信",
    tone: "neutral",
    detail: "",
  };
}

function PredictorOverviewCard({ overview }) {
  if (!overview) return null;
  const meta = overview.meta || {};
  const summaryRows = (overview.summaries || []).map((item) => ({
    モデル: item.predictor_label || item.predictor_id || "-",
    本命馬番: item.top_choice_horse_no || "-",
    本命馬名: item.top_choice_horse_name || "-",
    複勝圏確率: item.top_choice_top3_prob_model ?? "",
  }));
  const profileRows = (overview.profiles || []).map((item) => ({
    モデル: item.predictor_label || item.predictor_id || "-",
    利用可否: item.available ? "あり" : "なし",
  }));
  const performanceRows = overview.performance?.current_scope_history || [];

  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>予測モデル概要</h3>
        <span>{overview.current_context?.scope_label_ja || overview.current_context?.scope_key || "-"}</span>
      </div>
      <MetaRow
        items={[
          `利用可能 ${meta.available_predictor_count || 0}`,
          `本命分散 ${meta.unique_top1_count || 0}`,
          `合意軸 ${meta.consensus_top_horse_no || "-"}`,
        ]}
      />
      <DataTable title="合意状況" rows={overview.consensus || []} />
      <DataTable title="本命サマリー" rows={summaryRows} />
      <DataTable title="利用状況" rows={profileRows} />
      <DataTable title="対象範囲の履歴" rows={performanceRows} />
    </article>
  );
}

function PredictorCard({ item }) {
  const hasRows = item.top5_rows?.length || item.mark_rows?.length || item.summary_rows?.length;
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>{item.label || item.predictor_id || "-"}</h3>
        <span>{item.predictor_id || "-"}</span>
      </div>
      <DataTable title="上位5頭" rows={item.top5_rows} />
      <DataTable title="印" rows={item.mark_rows} />
      <DataTable title="概要" rows={item.summary_rows} />
      {!hasRows ? <p className="admin-workspace-card__empty">表示できる予測データはありません。</p> : null}
    </article>
  );
}

function PortfolioCard({ item }) {
  const today = item.today || {};
  const lookback = item.lookback_summary || {};
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>{item.engine_label || item.engine || "-"}</h3>
        <span>{item.engine || "-"}</span>
      </div>
      <DataTable
        title="当日残高"
        rows={[
          {
            日付: today.ledger_date || "-",
            利用可能額: today.available_bankroll_yen ?? "",
            未精算投資: today.open_stake_yen ?? "",
            実現損益: today.realized_profit_yen ?? "",
            追加入金: today.topup_yen ?? "",
          },
        ]}
        maxRows={1}
      />
      <DataTable
        title="14日集計"
        rows={[
          {
            期間: lookback.days ?? "",
            精算レース: lookback.settled_runs ?? "",
            精算買い目: lookback.settled_tickets ?? "",
            的中率: lookback.hit_rate ?? "",
            回収率: lookback.roi ?? "",
            損益: lookback.profit_yen ?? "",
          },
        ]}
        maxRows={1}
      />
      <DataTable title="直近日次" rows={item.recent_days || []} />
      <DataTable title="券種内訳" rows={item.bet_type_breakdown || []} />
      <DataTable title="直近買い目" rows={item.recent_tickets || []} />
    </article>
  );
}

function RunMetricsCard({ data }) {
  if (!data) return null;
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>実行集計</h3>
        <span>{data.run_id || "-"}</span>
      </div>
      <DataTable title="結果サマリー" rows={data.run_result_summary || []} />
      <DataTable title="券種サマリー" rows={data.run_bet_type_summary || []} />
      <DataTable title="買い目一覧" rows={data.run_bet_ticket_summary || []} />
      <DataTable title="予測結果サマリー" rows={data.run_predictor_summary || []} />
    </article>
  );
}

function RunContextCard({ contextRows, assetRows, recentRuns }) {
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>実行コンテキスト</h3>
        <span>概要</span>
      </div>
      <DataTable title="選択中の実行" rows={contextRows || []} maxRows={2} />
      <DataTable title="関連アセット" rows={assetRows || []} maxRows={12} />
      <DataTable title="直近実行" rows={recentRuns || []} maxRows={12} />
    </article>
  );
}

function OddsSnapshotCard({ data }) {
  if (!data) return null;
  const winRows = (data.win || []).map((item) => ({
    馬番: item.horse_no || "",
    馬名: item.name || "",
    オッズ: item.odds_raw || "",
  }));
  const placeRows = (data.place || []).map((item) => ({
    馬番: item.horse_no || "",
    馬名: item.name || "",
    オッズ: item.odds_raw || "",
  }));
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>オッズスナップショット</h3>
        <span>単勝 / 複勝</span>
      </div>
      <DataTable title="単勝オッズ" rows={winRows} maxRows={12} />
      <DataTable title="複勝オッズ" rows={placeRows} maxRows={12} />
    </article>
  );
}

function PolicyCard({ item }) {
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>{item.engine_label || item.engine || "-"}</h3>
        <span>{item.model || "-"}</span>
      </div>
      <MetaRow items={[`判定 ${item.bet_decision || "-"}`, `参加度 ${item.participation_level || "-"}`]} />
      <div className="admin-workspace-copy">
        <strong>{item.marks_text || "印なし"}</strong>
        <p>{item.ticket_plan_text || "買い目なし"}</p>
        <code>{item.result_triplet_text || "-"}</code>
      </div>
      {item.enabled_bet_types?.length ? <MetaRow items={[`券種 ${item.enabled_bet_types.join(", ")}`]} /> : null}
      {item.reason_codes?.length ? <MetaRow items={[`理由コード ${item.reason_codes.join(", ")}`]} /> : null}
      <DataTable title="買い目" rows={item.ticket_rows || []} />
      {item.payload_preview_text ? (
        <details className="admin-inline-panel">
          <summary>入力プレビュー</summary>
          <pre className="admin-workspace-output">{item.payload_preview_text}</pre>
        </details>
      ) : null}
    </article>
  );
}

function RecordPredictorForm({ busyKey, onSubmit }) {
  const [form, setForm] = useState({ top1: "", top2: "", top3: "" });

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <article className="admin-tool-panel">
      <div className="admin-tool-panel__body">
        <h3>予測結果の登録</h3>
        <p>選択中の実行に対して実際の上位3頭を登録します。</p>
        <form
          className="admin-inline-form"
          onSubmit={(event) => {
            event.preventDefault();
            onSubmit(form);
          }}
        >
          <label>
            <span>1着</span>
            <input value={form.top1} onChange={(event) => updateField("top1", event.target.value)} />
          </label>
          <label>
            <span>2着</span>
            <input value={form.top2} onChange={(event) => updateField("top2", event.target.value)} />
          </label>
          <label>
            <span>3着</span>
            <input value={form.top3} onChange={(event) => updateField("top3", event.target.value)} />
          </label>
          <div className="admin-inline-form__actions">
            <button type="submit" disabled={busyKey === "record_predictor"}>
              {busyKey === "record_predictor" ? "登録中..." : "登録"}
            </button>
          </div>
        </form>
      </div>
    </article>
  );
}

function WorkspaceOps({ busyKey, onAction }) {
  return (
    <section className="admin-tool-grid">
      <article className="admin-tool-panel">
        <div className="admin-tool-panel__body">
          <h3>単体LLM実行</h3>
          <p>選択したエンジンだけで買い目生成を実行します。</p>
          <div className="admin-toolbar">
            <button type="button" disabled={busyKey === "llm_gemini"} onClick={() => onAction("llm_gemini")}>
              {busyKey === "llm_gemini" ? "実行中..." : "Gemini"}
            </button>
            <button type="button" disabled={busyKey === "llm_deepseek"} onClick={() => onAction("llm_deepseek")}>
              {busyKey === "llm_deepseek" ? "実行中..." : "DeepSeek"}
            </button>
            <button type="button" disabled={busyKey === "llm_openai"} onClick={() => onAction("llm_openai")}>
              {busyKey === "llm_openai" ? "実行中..." : "OpenAI"}
            </button>
            <button type="button" disabled={busyKey === "llm_grok"} onClick={() => onAction("llm_grok")}>
              {busyKey === "llm_grok" ? "実行中..." : "Grok"}
            </button>
          </div>
        </div>
      </article>

      <article className="admin-tool-panel">
        <div className="admin-tool-panel__body">
          <h3>一括操作</h3>
          <p>全LLMの実行や当日資金の補充をまとめて行います。</p>
          <div className="admin-toolbar">
            <button type="button" disabled={busyKey === "llm_all"} onClick={() => onAction("llm_all")}>
              {busyKey === "llm_all" ? "実行中..." : "全LLM実行"}
            </button>
            <button type="button" disabled={busyKey === "topup"} onClick={() => onAction("topup")}>
              {busyKey === "topup" ? "反映中..." : "全LLM追加入金"}
            </button>
            <button type="button" disabled={busyKey === "fetch_result_and_settle"} onClick={() => onAction("fetch_result_and_settle")}>
              {busyKey === "fetch_result_and_settle" ? "精算中..." : "結果取得と精算"}
            </button>
          </div>
        </div>
      </article>
    </section>
  );
}

export default function AdminWorkspacePage({ appBasePath = "/keiba" }) {
  const search = new URLSearchParams(window.location.search);
  const initialScope = search.get("scope_key") || "central_dirt";
  const initialRunId = search.get("run_id") || "";

  const [token, setToken] = useState(() => window.sessionStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || "");
  const [scopeKey, setScopeKey] = useState(initialScope);
  const [runId, setRunId] = useState(initialRunId);
  const [reloadTick, setReloadTick] = useState(0);
  const [busyKey, setBusyKey] = useState("");
  const [opResult, setOpResult] = useState("");
  const [state, setState] = useState({ loading: false, error: "", data: null });

  useEffect(() => {
    if (!token.trim()) {
      setState({ loading: false, error: "", data: null });
      return;
    }
    let alive = true;
    setState({ loading: true, error: "", data: null });
    const params = new URLSearchParams();
    if (scopeKey) params.set("scope_key", scopeKey);
    if (runId) params.set("run_id", runId);
    fetch(`${appBasePath}/api/admin/workspace?${params.toString()}`, {
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token.trim()}`,
      },
    })
      .then((response) => {
        if (response.status === 403) throw new Error("管理トークンが無効です。");
        if (response.status === 404) throw new Error("実行IDが見つかりません。");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (!alive) return;
        setState({ loading: false, error: "", data });
        if ((!runId || runId !== data?.run_id) && data?.run_id) {
          setRunId(data.run_id);
          updateWorkspaceUrl(appBasePath, data.scope_key || scopeKey, data.run_id);
        }
      })
      .catch((error) => {
        if (!alive) return;
        if ((error?.message || "").includes("管理トークンが無効")) {
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          setToken("");
        }
        setState({ loading: false, error: error?.message || "ワークスペースの読み込みに失敗しました。", data: null });
      });
    return () => {
      alive = false;
    };
  }, [appBasePath, token, scopeKey, runId, reloadTick]);

  async function postWorkspaceAction(path, payload) {
    const response = await fetch(`${appBasePath}${path}`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        Authorization: `Bearer ${token.trim()}`,
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.ok === false) {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }
    return data;
  }

  async function handleWorkspaceAction(kind, extra = {}) {
    if (!state.data?.run_id || !state.data?.scope_key) return;
    setBusyKey(kind);
    setOpResult("");
    try {
      if (kind === "llm_all") {
        const data = await postWorkspaceAction("/api/admin/workspace/run_all_llm_buy", {
          scope_key: state.data.scope_key,
          run_id: state.data.run_id,
          refresh_odds: true,
        });
        const text = [
          ...(data.outputs || []).map((item) => `[${item.engine}]\n${item.output_text}`),
          ...(data.errors || []),
        ].join("\n\n");
        setOpResult(text);
      } else if (kind === "topup") {
        const data = await postWorkspaceAction("/api/admin/workspace/topup_all_llm_budget", {
          scope_key: state.data.scope_key,
          run_id: state.data.run_id,
        });
        setOpResult(
          [
            `ledger_date=${data.ledger_date}`,
            `amount_yen=${data.amount_yen}`,
            ...(data.summaries || []).map(
              (item) => `[${item.engine}] available=${item.available_bankroll_yen} topup=${item.topup_yen}`,
            ),
          ].join("\n"),
        );
      } else if (kind === "record_predictor") {
        const data = await postWorkspaceAction("/api/admin/workspace/record_predictor", {
          scope_key: state.data.scope_key,
          run_id: state.data.run_id,
          ...extra,
        });
        setOpResult(data.output_text || "");
      } else if (kind === "fetch_result_and_settle") {
        const data = await postWorkspaceAction("/api/admin/workspace/fetch_result_and_settle", {
          scope_key: state.data.scope_key,
          run_id: state.data.run_id,
        });
        setOpResult(
          [
            `job_id=${data.job_id || "-"}`,
            `race_id=${data.race_id || "-"}`,
            `actual_top3=${(data.actual_top3 || []).join(" / ")}`,
            data.result_url ? `result_url=${data.result_url}` : "",
            data.summary?.output || "",
          ]
            .filter(Boolean)
            .join("\n\n"),
        );
      } else {
        const engine = kind.replace("llm_", "");
        const data = await postWorkspaceAction("/api/admin/workspace/run_llm_buy", {
          scope_key: state.data.scope_key,
          run_id: state.data.run_id,
          policy_engine: engine,
          policy_model: "",
          refresh_odds: true,
        });
        setOpResult(data.output_text || "");
      }
      setReloadTick((value) => value + 1);
    } catch (error) {
      setState((prev) => ({ ...prev, error: error?.message || "ワークスペース操作に失敗しました。" }));
    } finally {
      setBusyKey("");
    }
  }

  const availableRuns = state.data?.available_runs || [];
  const availableScopes = state.data?.available_scopes || [];
  const predictorCards = state.data?.predictors || [];
  const predictorOverview = state.data?.predictor_overview || null;
  const policyCards = state.data?.policies || [];
  const portfolioSummaries = state.data?.portfolio_summaries || [];
  const oddsSnapshots = state.data?.odds_snapshots || null;
  const runContextRows = state.data?.run_context_rows || [];
  const runAssetRows = state.data?.run_asset_rows || [];
  const notify = notifyMeta(state.data);

  const summaryMeta = useMemo(() => {
    if (!state.data) return [];
    return [state.data.scope_label || state.data.scope_key || "-", state.data.run_id || "-", state.data.race_id || "-", state.data.race_date || "-"];
  }, [state.data]);

  if (!token.trim()) {
    return <AdminLoginPage appBasePath={appBasePath} redirectToLegacy={false} onAuthenticated={(nextToken) => setToken(nextToken)} />;
  }

  return (
    <main className="admin-jobs-page">
      <div className="admin-jobs-page__shell">
        <PageSectionHeader
          kicker="管理ワークスペース"
          title="実行ワークスペース"
          subtitle="予測モデル、LLM方針、資金サマリーを確認し、実行単位の操作をまとめて行えます。"
          meta={summaryMeta}
        />

        <section className="admin-toolbar">
          <label className="admin-toolbar__field">
            <span>対象区分</span>
            <select
              value={scopeKey}
              onChange={(event) => {
                const nextScope = event.target.value;
                setScopeKey(nextScope);
                setRunId("");
                setOpResult("");
                updateWorkspaceUrl(appBasePath, nextScope, "");
              }}
            >
              {availableScopes.map((item) => (
                <option key={item.scope_key} value={item.scope_key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="admin-toolbar__field admin-toolbar__field--wide">
            <span>実行ID</span>
            <select
              value={runId}
              onChange={(event) => {
                const nextRunId = event.target.value;
                setRunId(nextRunId);
                setOpResult("");
                updateWorkspaceUrl(appBasePath, scopeKey, nextRunId);
              }}
            >
              <option value="">最新を選択</option>
              {availableRuns.map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {item.label || item.run_id}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => setReloadTick((value) => value + 1)}>
            再読み込み
          </button>
          <a href={`${appBasePath}/console`}>タスク一覧</a>
        </section>

        {state.error ? <section className="notice-strip">{state.error}</section> : null}
        {!state.error && state.data?.ntfy_notify_status === "failed" && state.data?.ntfy_notify_error ? (
          <section className="notice-strip">{`ntfy 通知失敗: ${state.data.ntfy_notify_error}`}</section>
        ) : null}

        {state.loading ? (
          <section className="public-screen-state__panel">
            <span className="public-screen-state__eyebrow">読み込み中</span>
            <h1>ワークスペースを読み込んでいます</h1>
            <p>選択中の実行に対応する予測モデル、方針、資金データを取得しています。</p>
          </section>
        ) : null}

        {!state.loading && state.data ? (
          <>
            <section className="admin-summary-grid">
              <article className="admin-summary-card">
                <span>開催場</span>
                <strong>{state.data.location || "-"}</strong>
              </article>
              <article className="admin-summary-card">
                <span>実際の結果</span>
                <strong>
                  {[state.data.actual_result?.actual_top1, state.data.actual_result?.actual_top2, state.data.actual_result?.actual_top3]
                    .filter(Boolean)
                    .join(" / ") || "-"}
                </strong>
              </article>
              <article className="admin-summary-card admin-summary-card--good">
                <span>予測モデル</span>
                <strong>{predictorCards.length}</strong>
              </article>
              <article className="admin-summary-card admin-summary-card--active">
                <span>方針数</span>
                <strong>{policyCards.length}</strong>
              </article>
              <article className={`admin-summary-card${notify.tone === "good" ? " admin-summary-card--good" : notify.tone === "danger" ? " admin-summary-card--danger" : ""}`}>
                <span>ntfy</span>
                <strong>{notify.label}</strong>
                {notify.detail ? <small>{notify.detail}</small> : null}
              </article>
            </section>

            <section className="admin-workspace-grid">
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">構造化サマリー</h2>
                <PredictorOverviewCard overview={predictorOverview} />
                <RunContextCard contextRows={runContextRows} assetRows={runAssetRows} recentRuns={availableRuns} />
                <RunMetricsCard data={state.data} />
              </div>
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">LLM資金状況</h2>
                <OddsSnapshotCard data={oddsSnapshots} />
                {portfolioSummaries.length ? (
                  portfolioSummaries.map((item) => <PortfolioCard key={item.engine} item={item} />)
                ) : (
                  <section className="notice-strip">資金データはありません。</section>
                )}
              </div>
            </section>

            <WorkspaceOps busyKey={busyKey} onAction={handleWorkspaceAction} />
            <RecordPredictorForm busyKey={busyKey} onSubmit={(payload) => handleWorkspaceAction("record_predictor", payload)} />

            {opResult ? (
              <section className="notice-strip">
                <pre className="admin-workspace-output">{opResult}</pre>
              </section>
            ) : null}

            <section className="admin-workspace-grid">
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">予測モデル</h2>
                {predictorCards.length ? predictorCards.map((item) => <PredictorCard key={item.predictor_id} item={item} />) : <section className="notice-strip">予測データはありません。</section>}
              </div>
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">LLM方針</h2>
                {policyCards.length ? policyCards.map((item) => <PolicyCard key={item.engine} item={item} />) : <section className="notice-strip">方針データはありません。</section>}
              </div>
            </section>
          </>
        ) : null}
      </div>
    </main>
  );
}
