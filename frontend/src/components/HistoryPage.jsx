import React, { useState } from "react";

function TabButton({ active, onClick, children }) {
  return (
    <button type="button" className={active ? "is-active" : ""} onClick={onClick}>
      {children}
    </button>
  );
}

function EmptyState({ children }) {
  return <p className="history-empty-note">{children}</p>;
}

function OverviewCard({ label, value, note, accent = false }) {
  return (
    <article
      className={`history-overview-card${accent ? " history-overview-card--accent" : ""}`}
    >
      <span>{label}</span>
      <strong>{value || "-"}</strong>
      {note ? <p>{note}</p> : null}
    </article>
  );
}

function AgentHistoryTable({ title, eyebrow, columns, rows, rowKey }) {
  return (
    <section className="history-panel history-panel--agent">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
      </div>
      {rows.length ? (
        <div className="history-agent-table">
          <div className="history-agent-table__head">
            {columns.map((column) => (
              <span key={column.key}>{column.label}</span>
            ))}
          </div>
          {rows.map((item, index) => (
            <article key={rowKey(item, index)} className="history-agent-table__row">
              {columns.map((column) => (
                <span key={`${rowKey(item, index)}-${column.key}`}>
                  {column.render ? column.render(item) : item?.[column.key] || "-"}
                </span>
              ))}
            </article>
          ))}
        </div>
      ) : (
        <EmptyState>集計できる履歴データはまだありません。</EmptyState>
      )}
    </section>
  );
}

function AgentPredictionHistory({ data }) {
  const [periodKey, setPeriodKey] = useState("days_30");
  const agentHistory = data?.history?.agent_prediction || {};
  const periods = agentHistory?.periods || {};
  const periodTabs = [
    { key: "days_30", label: "月間" },
    { key: "days_365", label: "年間" },
    { key: "all_time", label: "累計" },
  ];
  const activePeriod = periods?.[periodKey] || {};
  const activePeriodLabel =
    periodTabs.find((item) => item.key === periodKey)?.label || "月間";
  const courseRows = Array.isArray(activePeriod?.course_rows) ? activePeriod.course_rows : [];
  const dailyRows = Array.isArray(activePeriod?.daily_rows) ? activePeriod.daily_rows : [];

  const courseColumns = [
    { key: "course", label: "場" },
    { key: "settled_races", label: "確定" },
    { key: "main_win_rate_text", label: "本命1着率" },
    { key: "main_top3_rate_text", label: "本命複勝圏率" },
    { key: "top5_cover_rate_text", label: "上位5頭カバー" },
  ];
  const dailyColumns = [
    { key: "race_date", label: "日付" },
    { key: "settled_races", label: "確定" },
    { key: "main_top3_rate_text", label: "本命複勝圏率" },
    { key: "top5_cover_rate_text", label: "上位5頭カバー" },
    { key: "top3_exact_rate_text", label: "上位3頭完全的中" },
  ];

  return (
    <section className="history-page">
      <div className="history-hero">
        <div className="history-hero__copy">
          <span className="history-hero__eyebrow">履歴分析</span>
          <h1>AI予測の命中率分析</h1>
          <p>
            公開したAI予測を結果と照合し、{activePeriodLabel}・年間・累計で本命精度と上位候補のカバー率を確認できます。
          </p>
        </div>

        <div className="history-hero__controls">
          <div className="history-period-tabs" role="tablist" aria-label="履歴期間">
            {periodTabs.map((item) => (
              <TabButton
                key={item.key}
                active={periodKey === item.key}
                onClick={() => setPeriodKey(item.key)}
              >
                {item.label}
              </TabButton>
            ))}
          </div>
        </div>
      </div>

      <div className="history-kpi-board history-kpi-board--agent">
        <OverviewCard
          label="本命1着率"
          value={activePeriod?.main_win_rate_text || "-"}
          note={`${activePeriod?.main_win_count || 0}/${activePeriod?.settled_races || 0}レース`}
          accent
        />
        <OverviewCard
          label="本命複勝圏率"
          value={activePeriod?.main_top3_rate_text || "-"}
          note={`${activePeriod?.main_top3_count || 0}/${activePeriod?.settled_races || 0}レース`}
        />
        <OverviewCard
          label="上位5頭カバー率"
          value={activePeriod?.top5_cover_rate_text || "-"}
          note={`${activePeriod?.top5_cover_hits || 0}/${(activePeriod?.settled_races || 0) * 3}頭`}
        />
        <OverviewCard
          label="予測レース"
          value={`${activePeriod?.predicted_races || 0}レース`}
          note={`結果確定 ${activePeriod?.settled_races || 0}レース`}
        />
        <OverviewCard
          label="高評価"
          value={`${activePeriod?.bet_races || 0}レース`}
          note={`見送り ${activePeriod?.skip_races || 0}レース`}
        />
        <OverviewCard
          label="上位3頭完全的中率"
          value={activePeriod?.top3_exact_rate_text || "-"}
          note={`${activePeriod?.top3_exact_count || 0}/${activePeriod?.settled_races || 0}レース`}
        />
      </div>

      <div className="history-table-layout">
        <AgentHistoryTable
          title="場別の命中傾向"
          eyebrow="場別集計"
          columns={courseColumns}
          rows={courseRows}
          rowKey={(item) => item?.course || "course"}
        />
        <AgentHistoryTable
          title="日別の予測成績"
          eyebrow="日別集計"
          columns={dailyColumns}
          rows={dailyRows}
          rowKey={(item) => item?.race_date || "date"}
        />
      </div>
    </section>
  );
}

export default function HistoryPage({ data, appShell = false }) {
  return <AgentPredictionHistory data={data} appShell={appShell} />;
}
