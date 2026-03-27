import React, { useEffect, useMemo, useState } from "react";
import AdminLoginPage, { ADMIN_TOKEN_STORAGE_KEY } from "./AdminLoginPage";
import PageSectionHeader from "./PageSectionHeader";

const TRACK_CONDITION_OPTIONS = [
  { value: "\u826f", label: "\u826f" },
  { value: "\u7a0d\u91cd", label: "\u7a0d\u91cd" },
  { value: "\u91cd", label: "\u91cd" },
  { value: "\u4e0d\u826f", label: "\u4e0d\u826f" },
];

const LOCATION_CANDIDATES = [
  "札幌",
  "函館",
  "福島",
  "新潟",
  "東京",
  "中山",
  "中京",
  "京都",
  "阪神",
  "小倉",
  "門別",
  "盛岡",
  "水沢",
  "浦和",
  "船橋",
  "大井",
  "川崎",
  "金沢",
  "笠松",
  "名古屋",
  "園田",
  "姫路",
  "高知",
  "佐賀",
];

function normalizeRaceMemoText(value) {
  return String(value || "")
    .replace(/[０-９]/g, (char) => String.fromCharCode(char.charCodeAt(0) - 65248))
    .replace(/：/g, ":")
    .replace(/／/g, "/")
    .replace(/（/g, "(")
    .replace(/）/g, ")")
    .replace(/\u3000/g, " ")
    .trim();
}

function parseRaceMemo(text, raceDate) {
  const raw = normalizeRaceMemoText(text);
  if (!raw) {
    return { updates: {}, message: "メモが空です。" };
  }

  const updates = {};
  let hitCount = 0;

  const timeMatch = raw.match(/(\d{1,2}:\d{2})\s*発走/);
  if (timeMatch?.[1] && String(raceDate || "").trim()) {
    updates.scheduled_off_time = `${String(raceDate).trim()}T${timeMatch[1]}`;
    hitCount += 1;
  }

  const distanceMatch = raw.match(/(?:芝|ダート|ダ|障害|障)\s*(\d{3,4})m/i);
  if (distanceMatch?.[1]) {
    updates.target_distance = distanceMatch[1];
    hitCount += 1;
  }

  const conditionMatch = raw.match(/馬場[:：]\s*(良|稍重|稍|重|不良|不)/);
  if (conditionMatch?.[1]) {
    const conditionMap = {
      良: "良",
      稍: "稍重",
      稍重: "稍重",
      重: "重",
      不: "不良",
      不良: "不良",
    };
    updates.target_track_condition = conditionMap[conditionMatch[1]] || conditionMatch[1];
    hitCount += 1;
  }

  const centralVenueMatch = raw.match(/\d+回\s*([^\s]+)\s+\d+日目/);
  if (centralVenueMatch?.[1]) {
    updates.location = centralVenueMatch[1];
    hitCount += 1;
  } else {
    const firstVenue = LOCATION_CANDIDATES.find((name) => raw.includes(name));
    if (firstVenue) {
      updates.location = firstVenue;
      hitCount += 1;
    }
  }

  const surfaceMatch = raw.match(/(芝|ダート|ダ|障害|障)\s*\d{3,4}m/i);
  const locationText = String(updates.location || "").trim();
  const isLocal = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀"].includes(locationText);
  if (surfaceMatch?.[1]) {
    const surface = surfaceMatch[1];
    if (isLocal) {
      updates.scope_key = "local";
      hitCount += 1;
    } else if (surface === "芝") {
      updates.scope_key = "central_turf";
      hitCount += 1;
    } else if (surface === "ダート" || surface === "ダ") {
      updates.scope_key = "central_dirt";
      hitCount += 1;
    }
  }

  if (!hitCount) {
    return { updates: {}, message: "メモから補完できるレース情報が見つかりませんでした。" };
  }

  return { updates, message: `メモから ${hitCount} 項目を補完しました。` };
}

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

function notifyMeta(job) {
  const status = String(job?.ntfy_notify_status || "").trim().toLowerCase();
  if (status === "notified") {
    return {
      label: `通知送信済み${job?.ntfy_notify_engine ? ` (${job.ntfy_notify_engine})` : ""}`,
      tone: "good",
    };
  }
  if (status === "failed") {
    return {
      label: `通知送信失敗${job?.ntfy_notify_error ? `: ${job.ntfy_notify_error}` : ""}`,
      tone: "danger",
    };
  }
  return {
    label: "通知未送信",
    tone: "muted",
  };
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
            <span>{entry.code !== "" ? `終了コード ${entry.code}` : "終了コード -"}</span>
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

function CreateJobForm({ onSubmit, busy, resetToken = 0 }) {
  const [form, setForm] = useState(createDefaultCreateJobForm);
  const [parseMessage, setParseMessage] = useState("");

  useEffect(() => {
    setForm(createDefaultCreateJobForm());
    setParseMessage("");
  }, [resetToken]);

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function applyParsedNotes() {
    const parsed = parseRaceMemo(form.notes, form.race_date);
    if (!Object.keys(parsed.updates || {}).length) {
      setParseMessage(parsed.message || "更新できる項目はありませんでした。");
      return;
    }
    setForm((prev) => ({ ...prev, ...parsed.updates }));
    setParseMessage(parsed.message || "メモから項目を補完しました。");
  }

  return (
    <details className="admin-tool-panel admin-tool-panel--primary" open>
      <summary>タスク作成</summary>
      <div className="admin-tool-panel__body admin-tool-panel__body--primary">
        <p>現在日時を初期値にして、素早くタスクを作成できます。</p>
      </div>
      <form
        className="admin-inline-form admin-inline-form--primary"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(form);
        }}
      >
        <label>
          <span>対象区分</span>
          <select value={form.scope_key} onChange={(event) => updateField("scope_key", event.target.value)}>
            <option value="central_dirt">中央ダート</option>
            <option value="central_turf">中央芝</option>
            <option value="local">地方</option>
          </select>
        </label>
        <label>
          <span>レースID</span>
          <input value={form.race_id} onChange={(event) => updateField("race_id", event.target.value)} />
        </label>
        <label>
          <span>開催場</span>
          <input value={form.location} onChange={(event) => updateField("location", event.target.value)} />
        </label>
        <label>
          <span>レース日</span>
          <input type="date" value={form.race_date} onChange={(event) => updateField("race_date", event.target.value)} />
        </label>
        <label>
          <span>発走時刻</span>
          <input
            type="datetime-local"
            step={300}
            value={form.scheduled_off_time}
            onChange={(event) => updateField("scheduled_off_time", event.target.value)}
          />
        </label>
        <label>
          <span>距離</span>
          <input type="number" step={100} value={form.target_distance} onChange={(event) => updateField("target_distance", event.target.value)} />
        </label>
        <label>
          <span>馬場</span>
          <TrackConditionSelect value={form.target_track_condition} onChange={(event) => updateField("target_track_condition", event.target.value)} />
        </label>
        <label>
          <span>前倒し分</span>
          <input type="number" value={form.lead_minutes} onChange={(event) => updateField("lead_minutes", event.target.value)} />
        </label>
        <label>
          <span>kachiuma.csv</span>
          <input
            key={`kachiuma-${resetToken}`}
            type="file"
            accept=".csv"
            onChange={(event) => updateField("kachiuma_file", event.target.files?.[0] || null)}
          />
        </label>
        <label>
          <span>shutuba.csv</span>
          <input
            key={`shutuba-${resetToken}`}
            type="file"
            accept=".csv"
            onChange={(event) => updateField("shutuba_file", event.target.files?.[0] || null)}
          />
        </label>
        <label className="admin-inline-form__wide">
          <span>メモ</span>
          <textarea rows={3} value={form.notes} onChange={(event) => updateField("notes", event.target.value)} />
        </label>
        <div className="admin-inline-form__actions">
          <button type="button" disabled={busy} onClick={applyParsedNotes}>
            メモから自動補完
          </button>
          {parseMessage ? <span>{parseMessage}</span> : null}
        </div>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "作成中..." : "作成"}
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
      <summary>アーカイブ取込</summary>
      <form
        className="admin-inline-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ archive_file: archiveFile, overwrite });
        }}
      >
        <label className="admin-inline-form__wide">
          <span>アーカイブZIP</span>
          <input type="file" accept=".zip,application/zip" onChange={(event) => setArchiveFile(event.target.files?.[0] || null)} />
        </label>
        <label className="admin-inline-form__wide admin-inline-form__checkbox">
          <input type="checkbox" checked={overwrite} onChange={(event) => setOverwrite(event.target.checked)} />
          <span>既存ファイルを上書き</span>
        </label>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "取込中..." : "取込"}
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
  const [parseMessage, setParseMessage] = useState("");

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

  function applyParsedNotes() {
    const parsed = parseRaceMemo(form.notes, form.race_date);
    if (!Object.keys(parsed.updates || {}).length) {
      setParseMessage(parsed.message || "更新できる項目はありませんでした。");
      return;
    }
    setForm((prev) => ({ ...prev, ...parsed.updates }));
    setParseMessage(parsed.message || "メモから項目を補完しました。");
  }

  return (
    <details className="admin-inline-panel">
      <summary>編集</summary>
      <form
        className="admin-inline-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(form);
        }}
      >
        <label>
          <span>レースID</span>
          <input value={form.race_id} onChange={(event) => updateField("race_id", event.target.value)} />
        </label>
        <label>
          <span>開催場</span>
          <input value={form.location} onChange={(event) => updateField("location", event.target.value)} />
        </label>
        <label>
          <span>レース日</span>
          <input type="date" value={form.race_date} onChange={(event) => updateField("race_date", event.target.value)} />
        </label>
        <label>
          <span>発走時刻</span>
          <input
            type="datetime-local"
            step={300}
            value={String(form.scheduled_off_time || "").slice(0, 16)}
            onChange={(event) => updateField("scheduled_off_time", event.target.value)}
          />
        </label>
        <label>
          <span>距離</span>
          <input type="number" step={100} value={form.target_distance} onChange={(event) => updateField("target_distance", event.target.value)} />
        </label>
        <label>
          <span>馬場</span>
          <TrackConditionSelect value={form.target_track_condition} onChange={(event) => updateField("target_track_condition", event.target.value)} />
        </label>
        <label>
          <span>前倒し分</span>
          <input type="number" value={form.lead_minutes} onChange={(event) => updateField("lead_minutes", event.target.value)} />
        </label>
        <label className="admin-inline-form__wide">
          <span>メモ</span>
          <textarea value={form.notes} onChange={(event) => updateField("notes", event.target.value)} rows={3} />
        </label>
        <div className="admin-inline-form__actions">
          <button type="button" disabled={busy} onClick={applyParsedNotes}>
            メモから自動補完
          </button>
          {parseMessage ? <span>{parseMessage}</span> : null}
        </div>
        <div className="admin-inline-form__actions">
          <button type="submit" disabled={busy}>
            {busy ? "保存中..." : "保存"}
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
        <h3>LLM状態リセット</h3>
        <p>復旧や調査でのみ使う低頻度のメンテナンス操作です。</p>
        <div className="admin-toolbar">
          <button type="button" disabled={busy} onClick={onReset}>
            {busy ? "リセット中..." : "LLM状態をリセット"}
          </button>
        </div>
      </div>
    </article>
  );
}

function JobCard({ job, onAction, busyAction }) {
  const actualText = [job.actual_top1, job.actual_top2, job.actual_top3].filter(Boolean).join(" / ") || "未確定";
  const title = `${job.location || ""}${job.race_id ? ` ${job.race_id}` : ""}`.trim() || job.job_id;
  const workspaceUrl =
    job.current_run_id && job.scope_key
      ? `/keiba/console/workspace?scope_key=${encodeURIComponent(job.scope_key)}&run_id=${encodeURIComponent(job.current_run_id)}`
      : "";
  const busy = busyAction === job.job_id;
  const notify = notifyMeta(job);

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
        <span>日付 {job.race_date || "-"}</span>
        <span>発走 {job.scheduled_off_time || "-"}</span>
        <span>実行ID {job.current_run_id || "-"}</span>
      </div>

      <div className="admin-job-card__steps">
        {(job.step_badges || []).map((item) => (
          <span key={item.step} className={stepClass(item.tone)}>
            {item.label}: {item.state_label}
          </span>
        ))}
      </div>

      <div className="admin-job-card__meta admin-job-card__meta--stack">
        <span>結果 {actualText}</span>
        <span className={statusClass(notify.tone)}>{notify.label}</span>
        {job.notes ? <span>メモ {job.notes}</span> : null}
      </div>

      <div className="admin-job-card__actions">
        <button type="button" disabled={busy} onClick={() => onAction("process_now", job)}>
          {busy ? "処理中..." : "今すぐ実行"}
        </button>
        <button type="button" disabled={busy} onClick={() => onAction("delete", job)}>
          削除
        </button>
        {workspaceUrl ? <a href={workspaceUrl}>ワークスペース</a> : null}
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
  const [createFormResetTick, setCreateFormResetTick] = useState(0);
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
          throw new Error("管理トークンが無効です。");
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
        if ((error?.message || "").includes("管理トークンが無効")) {
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          setToken("");
        }
        setState({ loading: false, error: error?.message || "管理タスクの読み込みに失敗しました。", data: null });
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
    if (kind === "delete" && !window.confirm(`タスク ${jobId} を削除しますか？`)) return;
    setBusyAction(jobId);
    setFlashMessage("");

    try {
      await postJson(`/api/admin/jobs/${kind}`, { job_id: jobId, ...payload });
      const messages = {
        process_now: `タスク ${jobId} の処理を開始しました。`,
        delete: `タスク ${jobId} を削除しました。`,
        edit: `タスク ${jobId} を更新しました。`,
      };
      setFlashMessage(messages[kind] || "操作が完了しました。");
      setReloadTick((value) => value + 1);
    } catch (error) {
      setState((prev) => ({ ...prev, error: error?.message || "操作に失敗しました。" }));
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
        setFlashMessage(`期限到来タスクを ${data.queued_count || 0} 件キューに追加しました。`);
      } else if (kind === "run_due_now") {
        setFlashMessage(`期限到来タスクを処理しました。実行 ${data.processed_count || 0} 件、精算 ${data.settled_count || 0} 件。`);
      } else if (kind === "topup_today_all_llm") {
        setFlashMessage(`全LLMの当日資金に ${data.amount_yen || 0} 円を追加しました。`);
      } else if (kind === "daily_summary_share") {
        const intentUrl = String(data?.intent_url || "").trim();
        if (intentUrl) {
          const popup = window.open(intentUrl, "_blank", "noopener,noreferrer");
          if (!popup) {
            window.location.href = intentUrl;
          }
        }
        setFlashMessage(`${data.target_date_label || data.target_date || "対象日"} の日次サマリー共有を開きました。`);
      } else if (kind === "generate_daily_report") {
        const publicUrl = String(data?.public_url || "").trim();
        if (publicUrl) {
          const popup = window.open(publicUrl, "_blank", "noopener,noreferrer");
          if (!popup) {
            window.location.href = publicUrl;
          }
        }
        setFlashMessage(`${data.target_date_label || data.target_date || "対象日"} の日報を保存しました。`);
      } else if (kind === "create") {
        setFlashMessage(`タスク ${data.job_id || ""} を作成しました。`);
        setCreateFormResetTick((value) => value + 1);
      } else if (kind === "import_archive") {
        setFlashMessage(`アーカイブを取り込みました。書き込み ${data.written || 0} 件、スキップ ${data.skipped || 0} 件。`);
      } else if (kind === "reset_llm_state") {
        setFlashMessage(data.output_text || "LLM状態のリセットが完了しました。");
      }
      setReloadTick((value) => value + 1);
    } catch (error) {
      setState((prev) => ({ ...prev, error: error?.message || "操作に失敗しました。" }));
    } finally {
      setBusyAction("");
    }
  }

  const summaryItems = useMemo(() => {
    const summary = state.data?.summary || {};
    return [
      { key: "total", label: "総数", value: summary.total || 0, tone: "neutral" },
      { key: "scheduled", label: "予定", value: summary.scheduled || 0, tone: "neutral" },
      { key: "processing", label: "処理中", value: summary.processing || 0, tone: "active" },
      { key: "ready", label: "準備完了", value: summary.ready || 0, tone: "good" },
      { key: "settled", label: "精算済み", value: summary.settled || 0, tone: "neutral" },
      { key: "failed", label: "失敗", value: summary.failed || 0, tone: "danger" },
    ];
  }, [state.data]);

  if (!token.trim()) {
    return <AdminLoginPage appBasePath={appBasePath} redirectToLegacy={false} onAuthenticated={(nextToken) => setToken(nextToken)} />;
  }

  return (
    <main className="admin-jobs-page">
      <div className="admin-jobs-page__shell">
        <PageSectionHeader
          kicker="管理タスク"
          title="タスク管理"
          subtitle="レースタスクの管理、処理実行、アーカイブ取込、保守操作をまとめて行えます。"
          meta={[`表示 ${(state.data?.jobs || []).length} 件`, showSettled ? "精算済みを表示" : "精算済みを非表示"]}
        />

        <section className="admin-toolbar">
          <button type="button" onClick={() => setShowSettled((value) => !value)}>
            {showSettled ? "精算済みを隠す" : "精算済みを表示"}
          </button>
          <button type="button" onClick={() => setReloadTick((value) => value + 1)}>
            再読み込み
          </button>
          <button type="button" disabled={busyAction === "scan_due"} onClick={() => runToolbarAction("scan_due", () => postJson("/api/admin/jobs/scan_due", {}))}>
            {busyAction === "scan_due" ? "確認中..." : "期限到来を確認"}
          </button>
          <button type="button" disabled={busyAction === "run_due_now"} onClick={() => runToolbarAction("run_due_now", () => postJson("/api/admin/jobs/run_due_now", {}))}>
            {busyAction === "run_due_now" ? "実行中..." : "期限到来を実行"}
          </button>
          <button
            type="button"
            disabled={busyAction === "topup_today_all_llm"}
            onClick={() => runToolbarAction("topup_today_all_llm", () => postJson("/api/admin/jobs/topup_today_all_llm", {}))}
          >
            {busyAction === "topup_today_all_llm" ? "反映中..." : "全LLM追加入金"}
          </button>
          <button
            type="button"
            disabled={busyAction === "daily_summary_share"}
            onClick={() => runToolbarAction("daily_summary_share", () => postJson("/api/admin/jobs/daily_summary_share", {}))}
          >
            {busyAction === "daily_summary_share" ? "準備中..." : "日次サマリー共有"}
          </button>
          <button
            type="button"
            disabled={busyAction === "generate_daily_report"}
            onClick={() => runToolbarAction("generate_daily_report", () => postJson("/api/admin/jobs/generate_daily_report", {}))}
          >
            {busyAction === "generate_daily_report" ? "生成中..." : "私の日報を生成"}
          </button>
        </section>

        <section className="admin-tool-hero">
          <CreateJobForm
            busy={busyAction === "create"}
            resetToken={createFormResetTick}
            onSubmit={(payload) => runToolbarAction("create", () => postForm("/api/admin/jobs/create", payload))}
          />
        </section>

        <details className="admin-tool-panel admin-tool-panel--secondary">
          <summary>補助ツール</summary>
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
            <span className="public-screen-state__eyebrow">読み込み中</span>
            <h1>タスクを読み込んでいます</h1>
            <p>最新のタスク一覧と状態サマリーを取得しています。</p>
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
            <span className="public-screen-state__eyebrow">空です</span>
            <h1>タスクはありません</h1>
            <p>タスク作成、アーカイブ取込、または期限到来確認から開始してください。</p>
          </section>
        ) : null}
      </div>
    </main>
  );
}
