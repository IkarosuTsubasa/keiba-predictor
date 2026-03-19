import React, { useEffect, useMemo, useState } from "react";
import AdminLoginPage, { ADMIN_TOKEN_STORAGE_KEY } from "./AdminLoginPage";
import PageSectionHeader from "./PageSectionHeader";

const TRACK_CONDITION_OPTIONS = [
  { value: "\u826f", label: "\u826f" },
  { value: "\u7a0d\u91cd", label: "\u7a0d\u91cd" },
  { value: "\u91cd", label: "\u91cd" },
  { value: "\u4e0d\u826f", label: "\u4e0d\u826f" },
];

function formatDateInputValue(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatDateTimeLocalValue(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function roundToFiveMinutes(date) {
  const rounded = new Date(date);
  rounded.setSeconds(0, 0);
  rounded.setMinutes(rounded.getMinutes() - (rounded.getMinutes() % 5));
  return rounded;
}

function createDefaultCreateJobForm() {
  const now = roundToFiveMinutes(new Date());
  return {
    scope_key: "local",
    race_id: "",
    location: "",
    race_date: formatDateInputValue(now),
    scheduled_off_time: formatDateTimeLocalValue(now),
    target_distance: "1600",
    target_track_condition: "\u826f",
    lead_minutes: "30",
    notes: "",
    kachiuma_file: null,
    shutuba_file: null,
  };
}

function statusClass(tone) {
  if (tone === "good") return "admin-job-card__status admin-job-card__status--good";
  if (tone === "danger") return "admin-job-card__status admin-job-card__status--danger";
  if (tone === "active") return "admin-job-card__status admin-job-card__status--active";
  return "admin-job-card__status";
}

function stepClass(tone) {
  if (tone === "good") return "admin-step-badge admin-step-badge--good";
  if (tone === "danger") return "admin-step-badge admin-step-badge--danger";
  if (tone === "active") return "admin-step-badge admin-step-badge--active";
  return "admin-step-badge";
}

function SummaryCard({ label, value, tone = "neutral" }) {
  return (
    <article className={`admin-summary-card admin-summary-card--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ProcessLog({ entries }) {
  if (!entries?.length) return null;
  return (
    <div className="admin-job-card__log">
      {entries.map((entry, index) => (
        <div key={`${entry.step}-${index}`} className="admin-job-card__log-item">
          <div className="admin-job-card__log-head">
            <strong>{entry.step || "-"}</strong>
            <span>{entry.code !== "" ? `exit ${entry.code}` : "exit -"}</span>
          </div>
          {entry.preview ? <p>{entry.preview}</p> : null}
        </div>
      ))}
    </div>
  );
}

function TrackConditionSelect({ value, onChange }) {
  return (
    <select value={value} onChange={onChange}>
      {TRACK_CONDITION_OPTIONS.map((item) => (
        <option key={item.value} value={item.value}>
          {item.label}
        </option>
      ))}
    </select>
  );
}

function CreateJobForm({ onSubmit, busy }) {
  const [form, setForm] = useState(createDefaultCreateJobForm);

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <details className="admin-tool-panel admin-tool-panel--primary" open>
      <summary>Create Task</summary>
      <div className="admin-tool-panel__body admin-tool-panel__body--primary">
        <p>Defaults to the current date and off time for fast task creation.</p>
      </div>
      <form
        className="admin-inline-form admin-inline-form--primary"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(form);
        }}
      >
        <label>
          <span>Scope</span>
          <select value={form.scope_key} onChange={(event) => updateField("scope_key", event.target.value)}>
            <option value="central_dirt">Central Dirt</option>
            <option value="central_turf">Central Turf</option>
            <option value="local">Local</option>
          </select>
        </label>
        <label>
          <span>Race ID</span>
          <input value={form.race_id} onChange={(event) => updateField("race_id", event.target.value)} />
        </label>
        <label>
          <span>Location</span>
          <input value={form.location} onChange={(event) => updateField("location", event.target.value)} />
        </label>
        <label>
          <span>Race Date</span>
          <input type="date" value={form.race_date} onChange={(event) => updateField("race_date", event.target.value)} />
        </label>
        <label>
          <span>Off Time</span>
          <input
            type="datetime-local"
            step={300}
            value={form.scheduled_off_time}
            onChange={(event) => updateField("scheduled_off_time", event.target.value)}
          />
        </label>
        <label>
          <span>Distance</span>
          <input type="number" step={100} value={form.target_distance} onChange={(event) => updateField("target_distance", event.target.value)} />
        </label>
        <label>
          <span>Track</span>
          <TrackConditionSelect value={form.target_track_condition} onChange={(event) => updateField("target_track_condition", event.target.value)} />
        </label>
        <label>
          <span>Lead Minutes</span>
          <input type="number" value={form.lead_minutes} onChange={(event) => updateField("lead_minutes", event.target.value)} />
        </label>
        <label>
          <span>kachiuma.csv</span>
          <input type="file" accept=".csv" onChange={(event) => updateField("kachiuma_file", event.target.files?.[0] || null)} />
        </label>
        <label>
          <span>shutuba.csv</span>
          <input type="file" accept=".csv" onChange={(event) => updateField("shutuba_file", event.target.files?.[0] || null)} />
        </label>
        <label className="admin-inline-form__wide">
          <span>Notes</span>
          <textarea rows={3} value={form.notes} onChange={(event) => updateField("notes", event.target.value)} />
        </label>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "Creating..." : "Create"}
          </button>
        </div>
      </form>
    </details>
  );
}

function ImportArchiveForm({ onSubmit, busy }) {
  const [archiveFile, setArchiveFile] = useState(null);
  const [overwrite, setOverwrite] = useState(false);

  return (
    <details className="admin-tool-panel">
      <summary>Import Archive</summary>
      <form
        className="admin-inline-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ archive_file: archiveFile, overwrite });
        }}
      >
        <label className="admin-inline-form__wide">
          <span>Archive ZIP</span>
          <input type="file" accept=".zip,application/zip" onChange={(event) => setArchiveFile(event.target.files?.[0] || null)} />
        </label>
        <label className="admin-inline-form__wide admin-inline-form__checkbox">
          <input type="checkbox" checked={overwrite} onChange={(event) => setOverwrite(event.target.checked)} />
          <span>Overwrite existing files</span>
        </label>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "Importing..." : "Import"}
          </button>
        </div>
      </form>
    </details>
  );
}

function EditJobForm({ job, onSubmit, busy }) {
  const [form, setForm] = useState({
    race_id: job.race_id || "",
    location: job.location || "",
    race_date: job.race_date || "",
    scheduled_off_time: job.scheduled_off_time || "",
    target_distance: job.target_distance || "",
    target_track_condition: job.target_track_condition || "\u826f",
    lead_minutes: job.lead_minutes || 30,
    notes: job.notes || "",
  });

  useEffect(() => {
    setForm({
      race_id: job.race_id || "",
      location: job.location || "",
      race_date: job.race_date || "",
      scheduled_off_time: job.scheduled_off_time || "",
      target_distance: job.target_distance || "",
      target_track_condition: job.target_track_condition || "\u826f",
      lead_minutes: job.lead_minutes || 30,
      notes: job.notes || "",
    });
  }, [job]);

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <details className="admin-inline-panel">
      <summary>Edit</summary>
      <form
        className="admin-inline-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(form);
        }}
      >
        <label>
          <span>Race ID</span>
          <input value={form.race_id} onChange={(event) => updateField("race_id", event.target.value)} />
        </label>
        <label>
          <span>Location</span>
          <input value={form.location} onChange={(event) => updateField("location", event.target.value)} />
        </label>
        <label>
          <span>Race Date</span>
          <input type="date" value={form.race_date} onChange={(event) => updateField("race_date", event.target.value)} />
        </label>
        <label>
          <span>Off Time</span>
          <input
            type="datetime-local"
            step={300}
            value={String(form.scheduled_off_time || "").slice(0, 16)}
            onChange={(event) => updateField("scheduled_off_time", event.target.value)}
          />
        </label>
        <label>
          <span>Distance</span>
          <input type="number" step={100} value={form.target_distance} onChange={(event) => updateField("target_distance", event.target.value)} />
        </label>
        <label>
          <span>Track</span>
          <TrackConditionSelect value={form.target_track_condition} onChange={(event) => updateField("target_track_condition", event.target.value)} />
        </label>
        <label>
          <span>Lead Minutes</span>
          <input type="number" value={form.lead_minutes} onChange={(event) => updateField("lead_minutes", event.target.value)} />
        </label>
        <label className="admin-inline-form__wide">
          <span>Notes</span>
          <textarea value={form.notes} onChange={(event) => updateField("notes", event.target.value)} rows={3} />
        </label>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "Saving..." : "Save"}
          </button>
        </div>
      </form>
    </details>
  );
}

function OpsPanel({ busy, onReset }) {
  return (
    <article className="admin-tool-panel admin-tool-panel--compact">
      <div className="admin-tool-panel__body">
        <h3>Reset LLM State</h3>
        <p>Low-frequency maintenance action for recovery and debugging only.</p>
        <div className="admin-toolbar">
          <button type="button" disabled={busy} onClick={onReset}>
            {busy ? "Resetting..." : "Reset LLM State"}
          </button>
        </div>
      </div>
    </article>
  );
}

function JobCard({ job, onAction, busyAction }) {
  const actualText = [job.actual_top1, job.actual_top2, job.actual_top3].filter(Boolean).join(" / ") || "Not settled";
  const title = `${job.location || ""}${job.race_id ? ` ${job.race_id}` : ""}`.trim() || job.job_id;
  const workspaceUrl =
    job.current_run_id && job.scope_key
      ? `/keiba/console/workspace?scope_key=${encodeURIComponent(job.scope_key)}&run_id=${encodeURIComponent(job.current_run_id)}`
      : "";
  const busy = busyAction === job.job_id;

  return (
    <article className="admin-job-card">
      <div className="admin-job-card__head">
        <div>
          <span className="admin-job-card__eyebrow">{job.scope_label || "-"}</span>
          <h3>{title}</h3>
        </div>
        <span className={statusClass(job.status_tone)}>{job.status_label || job.status || "-"}</span>
      </div>

      <div className="admin-job-card__meta">
        <span>Date {job.race_date || "-"}</span>
        <span>Off {job.scheduled_off_time || "-"}</span>
        <span>Run {job.current_run_id || "-"}</span>
      </div>

      <div className="admin-job-card__steps">
        {(job.step_badges || []).map((item) => (
          <span key={item.step} className={stepClass(item.tone)}>
            {item.label}: {item.state_label}
          </span>
        ))}
      </div>

      <div className="admin-job-card__meta admin-job-card__meta--stack">
        <span>Actual {actualText}</span>
        {job.notes ? <span>Notes {job.notes}</span> : null}
      </div>

      <div className="admin-job-card__actions">
        <button type="button" disabled={busy} onClick={() => onAction("process_now", job)}>
          {busy ? "Processing..." : "Process Now"}
        </button>
        <button type="button" disabled={busy} onClick={() => onAction("delete", job)}>
          Delete
        </button>
        {workspaceUrl ? <a href={workspaceUrl}>Workspace</a> : null}
      </div>

      <EditJobForm job={job} busy={busy} onSubmit={(payload) => onAction("edit", job, payload)} />
      <ProcessLog entries={job.process_log} />
    </article>
  );
}

export default function AdminJobsPage({ appBasePath = "/keiba" }) {
  const [token, setToken] = useState(() => window.sessionStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || "");
  const [showSettled, setShowSettled] = useState(false);
  const [reloadTick, setReloadTick] = useState(0);
  const [busyAction, setBusyAction] = useState("");
  const [flashMessage, setFlashMessage] = useState("");
  const [state, setState] = useState({
    loading: false,
    error: "",
    data: null,
  });

  useEffect(() => {
    if (!token.trim()) {
      setState({ loading: false, error: "", data: null });
      return;
    }

    let alive = true;
    setState((prev) => ({ ...prev, loading: true, error: "" }));
    fetch(`${appBasePath}/api/admin/jobs?show_settled=${showSettled ? "1" : "0"}`, {
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token.trim()}`,
      },
    })
      .then((response) => {
        if (response.status === 403) {
          throw new Error("Admin token invalid.");
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) return;
        setState({ loading: false, error: "", data });
      })
      .catch((error) => {
        if (!alive) return;
        if ((error?.message || "").includes("Admin token invalid")) {
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          setToken("");
        }
        setState({ loading: false, error: error?.message || "Failed to load admin jobs.", data: null });
      });

    return () => {
      alive = false;
    };
  }, [appBasePath, showSettled, token, reloadTick]);

  async function postJson(path, payload) {
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

  async function postForm(path, values) {
    const formData = new FormData();
    Object.entries(values).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      if (typeof File !== "undefined" && value instanceof File) {
        formData.append(key, value);
      } else if (typeof value === "boolean") {
        if (value) formData.append(key, "1");
      } else {
        formData.append(key, String(value));
      }
    });
    const response = await fetch(`${appBasePath}${path}`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token.trim()}`,
      },
      body: formData,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.ok === false) {
      throw new Error(data?.error || `HTTP ${response.status}`);
    }
    return data;
  }

  async function runAction(kind, job, payload = {}) {
    const jobId = String(job?.job_id || "").trim();
    if (!jobId) return;
    if (kind === "delete" && !window.confirm(`Delete task ${jobId}?`)) return;
    setBusyAction(jobId);
    setFlashMessage("");

    try {
      await postJson(`/api/admin/jobs/${kind}`, { job_id: jobId, ...payload });
      const messages = {
        process_now: `Task ${jobId} started.`,
        delete: `Task ${jobId} deleted.`,
        edit: `Task ${jobId} updated.`,
      };
      setFlashMessage(messages[kind] || "Action completed.");
      setReloadTick((value) => value + 1);
    } catch (error) {
      setState((prev) => ({ ...prev, error: error?.message || "Action failed." }));
    } finally {
      setBusyAction("");
    }
  }

  async function runToolbarAction(kind, action) {
    setBusyAction(kind);
    setFlashMessage("");
    try {
      const data = await action();
      if (kind === "scan_due") {
        setFlashMessage(`Queued ${data.queued_count || 0} due tasks.`);
      } else if (kind === "run_due_now") {
        setFlashMessage(`Processed due tasks. processed=${data.processed_count || 0}, settled=${data.settled_count || 0}`);
      } else if (kind === "topup_today_all_llm") {
        setFlashMessage(`Topped up all LLM bankrolls by ${data.amount_yen || 0} yen.`);
      } else if (kind === "create") {
        setFlashMessage(`Created task ${data.job_id || ""}.`);
      } else if (kind === "import_archive") {
        setFlashMessage(`Archive imported. written=${data.written || 0}, skipped=${data.skipped || 0}`);
      } else if (kind === "reset_llm_state") {
        setFlashMessage(data.output_text || "LLM state reset completed.");
      }
      setReloadTick((value) => value + 1);
    } catch (error) {
      setState((prev) => ({ ...prev, error: error?.message || "Action failed." }));
    } finally {
      setBusyAction("");
    }
  }

  const summaryItems = useMemo(() => {
    const summary = state.data?.summary || {};
    return [
      { key: "total", label: "Tasks", value: summary.total || 0, tone: "neutral" },
      { key: "scheduled", label: "Scheduled", value: summary.scheduled || 0, tone: "neutral" },
      { key: "processing", label: "Processing", value: summary.processing || 0, tone: "active" },
      { key: "ready", label: "Ready", value: summary.ready || 0, tone: "good" },
      { key: "settled", label: "Settled", value: summary.settled || 0, tone: "neutral" },
      { key: "failed", label: "Failed", value: summary.failed || 0, tone: "danger" },
    ];
  }, [state.data]);

  if (!token.trim()) {
    return <AdminLoginPage appBasePath={appBasePath} redirectToLegacy={false} onAuthenticated={(nextToken) => setToken(nextToken)} />;
  }

  return (
    <main className="admin-jobs-page">
      <div className="admin-jobs-page__shell">
        <PageSectionHeader
          kicker="Admin Jobs"
          title="Task Console"
          subtitle="Manage race tasks, trigger processing, import archives, and run maintenance actions."
          meta={[`${(state.data?.jobs || []).length} visible`, showSettled ? "showing settled" : "hiding settled"]}
        />

        <section className="admin-toolbar">
          <button type="button" onClick={() => setShowSettled((value) => !value)}>
            {showSettled ? "Hide Settled" : "Show Settled"}
          </button>
          <button type="button" onClick={() => setReloadTick((value) => value + 1)}>
            Refresh
          </button>
          <button type="button" disabled={busyAction === "scan_due"} onClick={() => runToolbarAction("scan_due", () => postJson("/api/admin/jobs/scan_due", {}))}>
            {busyAction === "scan_due" ? "Scanning..." : "Scan Due"}
          </button>
          <button type="button" disabled={busyAction === "run_due_now"} onClick={() => runToolbarAction("run_due_now", () => postJson("/api/admin/jobs/run_due_now", {}))}>
            {busyAction === "run_due_now" ? "Running..." : "Run Due Now"}
          </button>
          <button
            type="button"
            disabled={busyAction === "topup_today_all_llm"}
            onClick={() => runToolbarAction("topup_today_all_llm", () => postJson("/api/admin/jobs/topup_today_all_llm", {}))}
          >
            {busyAction === "topup_today_all_llm" ? "Applying..." : "Top Up All LLM"}
          </button>
        </section>

        <section className="admin-tool-hero">
          <CreateJobForm busy={busyAction === "create"} onSubmit={(payload) => runToolbarAction("create", () => postForm("/api/admin/jobs/create", payload))} />
        </section>

        <details className="admin-tool-panel admin-tool-panel--secondary">
          <summary>Rare Tools</summary>
          <div className="admin-tool-grid">
            <ImportArchiveForm busy={busyAction === "import_archive"} onSubmit={(payload) => runToolbarAction("import_archive", () => postForm("/api/admin/jobs/import_archive", payload))} />
            <OpsPanel busy={busyAction === "reset_llm_state"} onReset={() => runToolbarAction("reset_llm_state", () => postJson("/api/admin/ops/reset_llm_state", {}))} />
          </div>
        </details>

        {flashMessage ? <section className="notice-strip">{flashMessage}</section> : null}
        {state.error ? <section className="notice-strip">{state.error}</section> : null}

        <section className="admin-summary-grid">
          {summaryItems.map((item) => (
            <SummaryCard key={item.key} label={item.label} value={item.value} tone={item.tone} />
          ))}
        </section>

        {state.loading ? (
          <section className="public-screen-state__panel">
            <span className="public-screen-state__eyebrow">Loading</span>
            <h1>Loading tasks</h1>
            <p>Fetching the latest task list and status summary.</p>
          </section>
        ) : null}

        {!state.loading && state.data?.jobs?.length ? (
          <section className="admin-job-list">
            {state.data.jobs.map((job) => (
              <JobCard key={job.job_id} job={job} onAction={runAction} busyAction={busyAction} />
            ))}
          </section>
        ) : null}

        {!state.loading && !state.data?.jobs?.length ? (
          <section className="public-screen-state__panel">
            <span className="public-screen-state__eyebrow">Empty</span>
            <h1>No tasks</h1>
            <p>Create a task, import an archive, or scan due races to get started.</p>
          </section>
        ) : null}
      </div>
    </main>
  );
}
