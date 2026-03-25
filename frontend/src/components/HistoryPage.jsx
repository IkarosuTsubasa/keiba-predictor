import React, { useMemo, useState } from "react";

function parsePercentText(value) {
  const text = String(value || "").trim();
  const matched = text.match(/-?\d+(?:\.\d+)?/);
  if (!matched) return null;
  const number = Number(matched[0]);
  return Number.isFinite(number) ? number : null;
}

function formatYen(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "-";
  return `${new Intl.NumberFormat("ja-JP").format(number)}円`;
}

function sortByPercent(items, key) {
  return [...(items || [])].sort((left, right) => {
    const rightValue = parsePercentText(right?.[key]) ?? -9999;
    const leftValue = parsePercentText(left?.[key]) ?? -9999;
    return rightValue - leftValue;
  });
}

function rankLabel(index) {
  return String(index + 1).padStart(2, "0");
}

function metricWidth(value, base = 200) {
  const numeric = parsePercentText(value);
  if (!Number.isFinite(numeric)) return "8%";
  const clamped = Math.max(8, Math.min(100, (numeric / base) * 100));
  return `${clamped}%`;
}

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

function LlmHeroCard({ leader }) {
  if (!leader) {
    return (
      <article className="history-hero-leader">
        <div className="history-hero-leader__body">
          <strong>-</strong>
          <p>対象データがまだありません。</p>
        </div>
      </article>
    );
  }

  return (
    <article className="history-hero-leader">
      <div className="history-hero-leader__top">
        <span className="history-hero-leader__eyebrow">Top Return</span>
        <span className="history-hero-leader__badge">{leader.label}</span>
      </div>
      <div className="history-hero-leader__body">
        <strong>{leader.roi_text || "-"}</strong>
        <p>{`損益 ${formatYen(leader.profit_yen || 0)}`}</p>
      </div>
      <div className="history-hero-leader__meta">
        <span>{`${leader.runs || 0}レース`}</span>
        <span>{`投資 ${formatYen(leader.stake_yen || 0)}`}</span>
      </div>
    </article>
  );
}

function LlmRankBoard({ items, leader }) {
  return (
    <section className="history-panel history-panel--ranking">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">ランキング</span>
          <h2>LLM パフォーマンスランキング</h2>
        </div>
      </div>

      <div className="history-ranking-grid">
        <LlmHeroCard leader={leader} />

        <div className="history-rank-list">
          {items.length ? (
            items.map((item, index) => (
              <article key={item.engine || item.label} className="history-rank-item">
                <div className="history-rank-item__head">
                  <span className="history-rank-item__rank">{`Rank ${rankLabel(index)}`}</span>
                  <strong>{item.label || "-"}</strong>
                  <em>{item.roi_text || "-"}</em>
                </div>
                <div className="history-rank-item__bar">
                  <span style={{ width: metricWidth(item.roi_text) }} />
                </div>
                <div className="history-rank-item__meta">
                  <span>{`${item.runs || 0}レース`}</span>
                  <span>{`損益 ${formatYen(item.profit_yen || 0)}`}</span>
                </div>
              </article>
            ))
          ) : (
            <EmptyState>対象期間の LLM データはまだありません。</EmptyState>
          )}
        </div>
      </div>

      {items.length > 1 ? (
        <div className="history-compact-grid">
          {items.slice(1).map((item) => (
            <article key={`compact-${item.engine || item.label}`} className="history-compact-card">
              <span>{item.label || "-"}</span>
              <strong>{item.roi_text || "-"}</strong>
              <p>{`損益 ${formatYen(item.profit_yen || 0)}`}</p>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function LlmTrendTable({ items }) {
  return (
    <section className="history-panel">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">推移</span>
          <h2>日別の成績推移</h2>
        </div>
      </div>

      <div className="history-trend-table">
        {items.length ? (
          items.map((item) => (
            <article key={item.date} className="history-trend-row">
              <div className="history-trend-row__summary">
                <span>{item.date || "-"}</span>
                <strong>{item.roi_text || "-"}</strong>
                <em>{`損益 ${formatYen(item.profit_yen || 0)}`}</em>
              </div>
              <div className="history-trend-row__chips">
                {(item.cards || []).map((card) => (
                  <span key={`${item.date}-${card.engine || card.label}`}>
                    <b>{card.label || "-"}</b>
                    <strong>{card.roi_text || "-"}</strong>
                  </span>
                ))}
              </div>
            </article>
          ))
        ) : (
          <EmptyState>日別推移を表示できる履歴がまだありません。</EmptyState>
        )}
      </div>
    </section>
  );
}

function PredictorMatrix({ items }) {
  return (
    <section className="history-panel history-panel--predictor">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">比較表</span>
          <h2>量化モデル比較テーブル</h2>
        </div>
      </div>

      {items.length ? (
        <div className="history-predictor-table">
          <div className="history-predictor-table__head">
            <span>モデル</span>
            <span>Top1</span>
            <span>Top1複勝圏</span>
            <span>Top3</span>
            <span>Top3完全一致</span>
            <span>Top5→Top3</span>
            <span>Samples</span>
          </div>
          {items.map((item, index) => (
            <article
              key={item.predictor_id || item.label}
              className="history-predictor-table__row"
            >
              <div className="history-predictor-table__model">
                <span>{`Rank ${rankLabel(index)}`}</span>
                <strong>{item.label || "-"}</strong>
              </div>
              <span>{item.top1_hit_rate_text || "-"}</span>
              <span>{item.top1_in_top3_rate_text || "-"}</span>
              <span>{item.top3_hit_rate_text || "-"}</span>
              <span>{item.top3_exact_rate_text || "-"}</span>
              <span>{item.top5_to_top3_hit_rate_text || "-"}</span>
              <strong>{item.samples || 0}</strong>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState>量化モデルの履歴データはまだありません。</EmptyState>
      )}
    </section>
  );
}

function PredictorHighlightStrip({ top1Leader, top3Leader, exactLeader }) {
  return (
    <div className="history-highlight-strip">
      <OverviewCard
        label="Top1 最高"
        value={top1Leader?.label || "-"}
        note={top1Leader?.top1_hit_rate_text || "-"}
        accent
      />
      <OverviewCard
        label="Top3 最高"
        value={top3Leader?.label || "-"}
        note={top3Leader?.top3_hit_rate_text || "-"}
      />
      <OverviewCard
        label="完全一致 最高"
        value={exactLeader?.label || "-"}
        note={exactLeader?.top3_exact_rate_text || "-"}
      />
    </div>
  );
}

export default function HistoryPage({ data }) {
  const [groupKey, setGroupKey] = useState("llm");
  const [periodKey, setPeriodKey] = useState("days_30");

  const llmPeriods = data?.history?.llm?.periods || {};
  const predictorPeriods = data?.history?.predictor?.periods || {};

  const periodTabs = [
    { key: "days_30", label: llmPeriods?.days_30?.label || "30日" },
    { key: "days_365", label: llmPeriods?.days_365?.label || "365日" },
    { key: "all_time", label: llmPeriods?.all_time?.label || "累計" },
  ];

  const activePeriodLabel =
    periodTabs.find((item) => item.key === periodKey)?.label || "30日";

  const llmPeriod = llmPeriods?.[periodKey] || { cards: [], totals: {}, trend: [] };
  const predictorPeriod = predictorPeriods?.[periodKey] || { cards: [], totals: {} };

  const llmCards = Array.isArray(llmPeriod.cards) ? llmPeriod.cards : [];
  const llmTrend = Array.isArray(llmPeriod.trend) ? llmPeriod.trend : [];
  const predictorCards = Array.isArray(predictorPeriod.cards)
    ? predictorPeriod.cards
    : [];

  const rankedLlmCards = useMemo(
    () => sortByPercent(llmCards, "roi_text"),
    [llmCards],
  );
  const rankedPredictorCards = useMemo(
    () => sortByPercent(predictorCards, "top1_hit_rate_text"),
    [predictorCards],
  );

  const llmLeader = rankedLlmCards[0] || null;
  const predictorLeaders = useMemo(
    () => ({
      top1: rankedPredictorCards[0] || null,
      top3: sortByPercent(predictorCards, "top3_hit_rate_text")[0] || null,
      exact: sortByPercent(predictorCards, "top3_exact_rate_text")[0] || null,
    }),
    [predictorCards, rankedPredictorCards],
  );

  const llmOverview = [
    {
      label: "総合回収率",
      value: llmPeriod?.totals?.roi_text || "-",
      note: `損益 ${formatYen(llmPeriod?.totals?.profit_yen || 0)}`,
      accent: true,
    },
    {
      label: "トップモデル",
      value: llmLeader?.label || "-",
      note: llmLeader?.roi_text || "-",
      accent: false,
    },
    {
      label: "対象レース数",
      value: `${llmPeriod?.totals?.runs || 0}レース`,
      note: `${activePeriodLabel}の集計`,
      accent: false,
    },
  ];

  const predictorOverview = [
    {
      label: "総サンプル",
      value: String(predictorPeriod?.totals?.samples || 0),
      note: `${activePeriodLabel}の総対象数`,
      accent: true,
    },
    {
      label: "対象モデル数",
      value: String(predictorCards.length),
      note: "Predictor V1-V5",
      accent: false,
    },
    {
      label: "分析期間",
      value: activePeriodLabel,
      note: "長期スパンで比較",
      accent: false,
    },
  ];

  return (
    <section className="history-page">
      <div className="history-hero">
        <div className="history-hero__copy">
          <span className="history-hero__eyebrow">履歴分析</span>
          <h1>長期の結果から、モデルの強さを比較する</h1>
          <p>
            30日、365日、累計の各スパンで回収率と命中率の差を確認できる、
            公開向けの履歴分析ページです。
          </p>
        </div>

        <div className="history-hero__controls">
          <div className="history-hero__tabs" role="tablist" aria-label="履歴グループ">
            <TabButton active={groupKey === "llm"} onClick={() => setGroupKey("llm")}>
              LLM 履歴
            </TabButton>
            <TabButton
              active={groupKey === "predictor"}
              onClick={() => setGroupKey("predictor")}
            >
              量化モデル
            </TabButton>
          </div>
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

      <div className="history-overview-grid">
        {(groupKey === "llm" ? llmOverview : predictorOverview).map((item) => (
          <OverviewCard
            key={item.label}
            label={item.label}
            value={item.value}
            note={item.note}
            accent={item.accent}
          />
        ))}
      </div>

      {groupKey === "llm" ? (
        <>
          <LlmRankBoard items={rankedLlmCards} leader={llmLeader} />
          <LlmTrendTable items={llmTrend} />
        </>
      ) : (
        <>
          <PredictorHighlightStrip
            top1Leader={predictorLeaders.top1}
            top3Leader={predictorLeaders.top3}
            exactLeader={predictorLeaders.exact}
          />
          <PredictorMatrix items={rankedPredictorCards} />
        </>
      )}
    </section>
  );
}
