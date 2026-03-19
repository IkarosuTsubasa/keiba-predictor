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

function PredictorOverviewCard({ overview }) {
  if (!overview) return null;
  const meta = overview.meta || {};
  const summaryRows = (overview.summaries || []).map((item) => ({
    predictor: item.predictor_label || item.predictor_id || "-",
    top1: item.top_choice_horse_no || "-",
    horse: item.top_choice_horse_name || "-",
    top3_prob: item.top_choice_top3_prob_model ?? "",
  }));
  const profileRows = (overview.profiles || []).map((item) => ({
    predictor: item.predictor_label || item.predictor_id || "-",
    available: item.available ? "yes" : "no",
  }));
  const performanceRows = overview.performance?.current_scope_history || [];

  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>Predictor Overview</h3>
        <span>{overview.current_context?.scope_label_ja || overview.current_context?.scope_key || "-"}</span>
      </div>
      <MetaRow
        items={[
          `predictors ${meta.available_predictor_count || 0}`,
          `unique top1 ${meta.unique_top1_count || 0}`,
          `consensus ${meta.consensus_top_horse_no || "-"}`,
        ]}
      />
      <DataTable title="Consensus" rows={overview.consensus || []} />
      <DataTable title="Top Choice Summary" rows={summaryRows} />
      <DataTable title="Availability" rows={profileRows} />
      <DataTable title="Scope History" rows={performanceRows} />
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
      <DataTable title="Top5" rows={item.top5_rows} />
      <DataTable title="Marks" rows={item.mark_rows} />
      <DataTable title="Summary" rows={item.summary_rows} />
      {!hasRows ? <p className="admin-workspace-card__empty">No predictor rows.</p> : null}
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
        title="Today"
        rows={[
          {
            ledger_date: today.ledger_date || "-",
            available_yen: today.available_bankroll_yen ?? "",
            open_stake_yen: today.open_stake_yen ?? "",
            realized_profit_yen: today.realized_profit_yen ?? "",
            topup_yen: today.topup_yen ?? "",
          },
        ]}
        maxRows={1}
      />
      <DataTable
        title="14-Day Summary"
        rows={[
          {
            days: lookback.days ?? "",
            settled_runs: lookback.settled_runs ?? "",
            settled_tickets: lookback.settled_tickets ?? "",
            hit_rate: lookback.hit_rate ?? "",
            roi: lookback.roi ?? "",
            profit_yen: lookback.profit_yen ?? "",
          },
        ]}
        maxRows={1}
      />
      <DataTable title="Recent Days" rows={item.recent_days || []} />
      <DataTable title="Bet Type Breakdown" rows={item.bet_type_breakdown || []} />
      <DataTable title="Recent Tickets" rows={item.recent_tickets || []} />
    </article>
  );
}

function RunMetricsCard({ data }) {
  if (!data) return null;
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>Run Metrics</h3>
        <span>{data.run_id || "-"}</span>
      </div>
      <DataTable title="Result Summary" rows={data.run_result_summary || []} />
      <DataTable title="Bet Type Summary" rows={data.run_bet_type_summary || []} />
      <DataTable title="Bet Tickets" rows={data.run_bet_ticket_summary || []} />
      <DataTable title="Predictor Result Summary" rows={data.run_predictor_summary || []} />
    </article>
  );
}

function RunContextCard({ contextRows, assetRows, recentRuns }) {
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>Run Context</h3>
        <span>summary</span>
      </div>
      <DataTable title="Selected Run" rows={contextRows || []} maxRows={2} />
      <DataTable title="Run Assets" rows={assetRows || []} maxRows={12} />
      <DataTable title="Recent Runs" rows={recentRuns || []} maxRows={12} />
    </article>
  );
}

function OddsSnapshotCard({ data }) {
  if (!data) return null;
  const winRows = (data.win || []).map((item) => ({
    horse_no: item.horse_no || "",
    name: item.name || "",
    odds: item.odds_raw || "",
  }));
  const placeRows = (data.place || []).map((item) => ({
    horse_no: item.horse_no || "",
    name: item.name || "",
    odds: item.odds_raw || "",
  }));
  return (
    <article className="admin-workspace-card">
      <div className="admin-workspace-card__head">
        <h3>Odds Snapshot</h3>
        <span>win / place</span>
      </div>
      <DataTable title="Win Odds" rows={winRows} maxRows={12} />
      <DataTable title="Place Odds" rows={placeRows} maxRows={12} />
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
      <MetaRow items={[`decision ${item.bet_decision || "-"}`, `level ${item.participation_level || "-"}`]} />
      <div className="admin-workspace-copy">
        <strong>{item.marks_text || "No marks"}</strong>
        <p>{item.ticket_plan_text || "No ticket plan"}</p>
        <code>{item.result_triplet_text || "-"}</code>
      </div>
      {item.enabled_bet_types?.length ? <MetaRow items={[`bet types ${item.enabled_bet_types.join(", ")}`]} /> : null}
      {item.reason_codes?.length ? <MetaRow items={[`reason codes ${item.reason_codes.join(", ")}`]} /> : null}
      <DataTable title="Tickets" rows={item.ticket_rows || []} />
      {item.payload_preview_text ? (
        <details className="admin-inline-panel">
          <summary>Payload Preview</summary>
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
        <h3>Record Predictor</h3>
        <p>Submit the actual top3 result for the selected run.</p>
        <form
          className="admin-inline-form"
          onSubmit={(event) => {
            event.preventDefault();
            onSubmit(form);
          }}
        >
          <label>
            <span>Top1</span>
            <input value={form.top1} onChange={(event) => updateField("top1", event.target.value)} />
          </label>
          <label>
            <span>Top2</span>
            <input value={form.top2} onChange={(event) => updateField("top2", event.target.value)} />
          </label>
          <label>
            <span>Top3</span>
            <input value={form.top3} onChange={(event) => updateField("top3", event.target.value)} />
          </label>
          <div className="admin-inline-form__actions">
            <button type="submit" disabled={busyKey === "record_predictor"}>
              {busyKey === "record_predictor" ? "Running..." : "Record"}
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
          <h3>Run Single LLM</h3>
          <p>Run policy buy for one selected engine.</p>
          <div className="admin-toolbar">
            <button type="button" disabled={busyKey === "llm_gemini"} onClick={() => onAction("llm_gemini")}>
              {busyKey === "llm_gemini" ? "Running..." : "Gemini"}
            </button>
            <button type="button" disabled={busyKey === "llm_deepseek"} onClick={() => onAction("llm_deepseek")}>
              {busyKey === "llm_deepseek" ? "Running..." : "DeepSeek"}
            </button>
            <button type="button" disabled={busyKey === "llm_openai"} onClick={() => onAction("llm_openai")}>
              {busyKey === "llm_openai" ? "Running..." : "OpenAI"}
            </button>
            <button type="button" disabled={busyKey === "llm_grok"} onClick={() => onAction("llm_grok")}>
              {busyKey === "llm_grok" ? "Running..." : "Grok"}
            </button>
          </div>
        </div>
      </article>

      <article className="admin-tool-panel">
        <div className="admin-tool-panel__body">
          <h3>Batch Actions</h3>
          <p>Run all LLM engines or top up the daily bankroll.</p>
          <div className="admin-toolbar">
            <button type="button" disabled={busyKey === "llm_all"} onClick={() => onAction("llm_all")}>
              {busyKey === "llm_all" ? "Running..." : "Run All LLM"}
            </button>
            <button type="button" disabled={busyKey === "topup"} onClick={() => onAction("topup")}>
              {busyKey === "topup" ? "Applying..." : "Top Up All LLM"}
            </button>
            <button type="button" disabled={busyKey === "fetch_result_and_settle"} onClick={() => onAction("fetch_result_and_settle")}>
              {busyKey === "fetch_result_and_settle" ? "Settling..." : "Fetch Result & Settle"}
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
        if (response.status === 403) throw new Error("Admin token invalid.");
        if (response.status === 404) throw new Error("Run not found.");
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
        if ((error?.message || "").includes("Admin token invalid")) {
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          setToken("");
        }
        setState({ loading: false, error: error?.message || "Failed to load workspace.", data: null });
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
      setState((prev) => ({ ...prev, error: error?.message || "Workspace action failed." }));
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
          kicker="Workspace"
          title="Run Workspace"
          subtitle="Inspect predictors, LLM policies, portfolio summaries, and execute run-level operations from one screen."
          meta={summaryMeta}
        />

        <section className="admin-toolbar">
          <label className="admin-toolbar__field">
            <span>Scope</span>
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
            <span>Run</span>
            <select
              value={runId}
              onChange={(event) => {
                const nextRunId = event.target.value;
                setRunId(nextRunId);
                setOpResult("");
                updateWorkspaceUrl(appBasePath, scopeKey, nextRunId);
              }}
            >
              <option value="">Select latest run</option>
              {availableRuns.map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {item.label || item.run_id}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => setReloadTick((value) => value + 1)}>
            Refresh
          </button>
          <a href={`${appBasePath}/console`}>Task List</a>
        </section>

        {state.error ? <section className="notice-strip">{state.error}</section> : null}

        {state.loading ? (
          <section className="public-screen-state__panel">
            <span className="public-screen-state__eyebrow">Loading</span>
            <h1>Loading workspace</h1>
            <p>Fetching predictor, policy, and portfolio data for the selected run.</p>
          </section>
        ) : null}

        {!state.loading && state.data ? (
          <>
            <section className="admin-summary-grid">
              <article className="admin-summary-card">
                <span>Location</span>
                <strong>{state.data.location || "-"}</strong>
              </article>
              <article className="admin-summary-card">
                <span>Actual Result</span>
                <strong>
                  {[state.data.actual_result?.actual_top1, state.data.actual_result?.actual_top2, state.data.actual_result?.actual_top3]
                    .filter(Boolean)
                    .join(" / ") || "-"}
                </strong>
              </article>
              <article className="admin-summary-card admin-summary-card--good">
                <span>Predictors</span>
                <strong>{predictorCards.length}</strong>
              </article>
              <article className="admin-summary-card admin-summary-card--active">
                <span>Policies</span>
                <strong>{policyCards.length}</strong>
              </article>
            </section>

            <section className="admin-workspace-grid">
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">Structured Overview</h2>
                <PredictorOverviewCard overview={predictorOverview} />
                <RunContextCard contextRows={runContextRows} assetRows={runAssetRows} recentRuns={availableRuns} />
                <RunMetricsCard data={state.data} />
              </div>
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">LLM Portfolio</h2>
                <OddsSnapshotCard data={oddsSnapshots} />
                {portfolioSummaries.length ? (
                  portfolioSummaries.map((item) => <PortfolioCard key={item.engine} item={item} />)
                ) : (
                  <section className="notice-strip">No portfolio data.</section>
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
                <h2 className="admin-workspace-section-title">Predictors</h2>
                {predictorCards.length ? predictorCards.map((item) => <PredictorCard key={item.predictor_id} item={item} />) : <section className="notice-strip">No predictor data.</section>}
              </div>
              <div className="admin-workspace-column">
                <h2 className="admin-workspace-section-title">Policies</h2>
                {policyCards.length ? policyCards.map((item) => <PolicyCard key={item.engine} item={item} />) : <section className="notice-strip">No policy data.</section>}
              </div>
            </section>
          </>
        ) : null}
      </div>
    </main>
  );
}
