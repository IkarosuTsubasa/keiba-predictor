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
  return `Rank ${String(index + 1).padStart(2, "0")}`;
}

function metricWidth(value, base = 200) {
  const numeric = parsePercentText(value);
  if (!Number.isFinite(numeric)) return "8%";
  const clamped = Math.max(8, Math.min(100, (numeric / base) * 100));
  return `${clamped}%`;
}

function pickDailyLeader(cards) {
  return sortByPercent(cards, "roi_text")[0] || null;
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

function HistoryStatPill({ label, value }) {
  return (
    <div className="history-stat-pill">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function LlmLeaderCard({ leader }) {
  if (!leader) {
    return (
      <article className="history-hero-leader">
        <div className="history-hero-leader__body">
          <strong>-</strong>
          <p>対象期間の LLM データはまだありません。</p>
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

function LlmRankingPanel({ items }) {
  const leader = items[0] || null;

  return (
    <section className="history-panel history-panel--ranking">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">ランキング</span>
          <h2>LLM パフォーマンスランキング</h2>
        </div>
      </div>

      <div className="history-ranking-grid">
        <LlmLeaderCard leader={leader} />

        <div className="history-rank-list">
          {items.length ? (
            items.map((item, index) => (
              <article key={item.engine || item.label} className="history-rank-item">
                <div className="history-rank-item__head">
                  <span className="history-rank-item__rank">{rankLabel(index)}</span>
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
            <EmptyState>対象期間のランキングを表示できるデータがありません。</EmptyState>
          )}
        </div>
      </div>
    </section>
  );
}

function HistoryArchivePanel({ items }) {
  return (
    <section className="history-panel history-panel--archive">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">アーカイブ</span>
          <h2>日別の結果一覧</h2>
        </div>
      </div>

      {items.length ? (
        <div className="history-archive-table">
          <div className="history-archive-table__head">
            <span>日付</span>
            <span>トップモデル</span>
            <span>対象数</span>
            <span>損益</span>
            <span>回収率</span>
          </div>
          {items.map((item) => {
            const leader = pickDailyLeader(item.cards || []);
            return (
              <article key={item.date} className="history-archive-table__row">
                <span>{item.date || "-"}</span>
                <div className="history-archive-table__model">
                  <strong>{leader?.label || "集計中"}</strong>
                  <em>{leader?.roi_text || "-"}</em>
                </div>
                <span>{`${(item.cards || []).length}モデル`}</span>
                <strong>{formatYen(item.profit_yen || 0)}</strong>
                <strong>{item.roi_text || "-"}</strong>
              </article>
            );
          })}
        </div>
      ) : (
        <EmptyState>日別アーカイブはまだありません。</EmptyState>
      )}
    </section>
  );
}

function PredictorHighlightStrip({ leaders }) {
  return (
    <div className="history-highlight-strip">
      <OverviewCard
        label="Top1 最高"
        value={leaders.top1?.label || "-"}
        note={leaders.top1?.top1_hit_rate_text || "-"}
        accent
      />
      <OverviewCard
        label="Top3 最高"
        value={leaders.top3?.label || "-"}
        note={leaders.top3?.top3_hit_rate_text || "-"}
      />
      <OverviewCard
        label="完全一致 最高"
        value={leaders.exact?.label || "-"}
        note={leaders.exact?.top3_exact_rate_text || "-"}
      />
    </div>
  );
}

function PredictorDesk({ items }) {
  return (
    <section className="history-panel history-panel--predictor">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">比較表</span>
          <h2>量化モデル比較テーブル</h2>
        </div>
      </div>

      {items.length ? (
        <>
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
                  <span>{rankLabel(index)}</span>
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

          <div className="history-predictor-card-grid">
            {items.slice(0, 3).map((item, index) => (
              <article
                key={`predictor-card-${item.predictor_id || item.label}`}
                className="history-predictor-card"
              >
                <span className="history-predictor-card__rank">{rankLabel(index)}</span>
                <h3>{item.label || "-"}</h3>
                <div className="history-predictor-card__stats">
                  <HistoryStatPill label="Top1" value={item.top1_hit_rate_text || "-"} />
                  <HistoryStatPill label="Top3" value={item.top3_hit_rate_text || "-"} />
                  <HistoryStatPill
                    label="完全一致"
                    value={item.top3_exact_rate_text || "-"}
                  />
                </div>
              </article>
            ))}
          </div>
        </>
      ) : (
        <EmptyState>量化モデルの履歴データはまだありません。</EmptyState>
      )}
    </section>
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

  const predictorLeaders = useMemo(
    () => ({
      top1: rankedPredictorCards[0] || null,
      top3: sortByPercent(predictorCards, "top3_hit_rate_text")[0] || null,
      exact: sortByPercent(predictorCards, "top3_exact_rate_text")[0] || null,
    }),
    [predictorCards, rankedPredictorCards],
  );

  const heroStats =
    groupKey === "llm"
      ? [
          {
            label: "総合回収率",
            value: llmPeriod?.totals?.roi_text || "-",
          },
          {
            label: "損益",
            value: formatYen(llmPeriod?.totals?.profit_yen || 0),
          },
          {
            label: "対象レース",
            value: `${llmPeriod?.totals?.runs || 0}レース`,
          },
        ]
      : [
          {
            label: "総サンプル",
            value: String(predictorPeriod?.totals?.samples || 0),
          },
          {
            label: "対象モデル",
            value: `${predictorCards.length}モデル`,
          },
          {
            label: "分析期間",
            value: activePeriodLabel,
          },
        ];

  const overviewCards =
    groupKey === "llm"
      ? [
          {
            label: "総合回収率",
            value: llmPeriod?.totals?.roi_text || "-",
            note: `損益 ${formatYen(llmPeriod?.totals?.profit_yen || 0)}`,
            accent: true,
          },
          {
            label: "トップモデル",
            value: rankedLlmCards[0]?.label || "-",
            note: rankedLlmCards[0]?.roi_text || "-",
            accent: false,
          },
          {
            label: "対象レース数",
            value: `${llmPeriod?.totals?.runs || 0}レース`,
            note: `${activePeriodLabel}の集計`,
            accent: false,
          },
        ]
      : [
          {
            label: "総サンプル",
            value: String(predictorPeriod?.totals?.samples || 0),
            note: `${activePeriodLabel}の総対象数`,
            accent: true,
          },
          {
            label: "最高 Top1",
            value: predictorLeaders.top1?.label || "-",
            note: predictorLeaders.top1?.top1_hit_rate_text || "-",
            accent: false,
          },
          {
            label: "対象モデル数",
            value: `${predictorCards.length}モデル`,
            note: "Predictor V1-V5",
            accent: false,
          },
        ];

  return (
    <section className="history-page">
      <div className="history-hero">
        <div className="history-hero__copy">
          <span className="history-hero__eyebrow">履歴分析</span>
          <h1>長期の結果から、モデルの強さを見比べる</h1>
          <p>
            30日、365日、累計の各スパンで、公開モデルと量化モデルの回収率、
            命中率、損益の差を比較できる分析ページです。
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
          <div className="history-hero__stat-row">
            {heroStats.map((item) => (
              <HistoryStatPill key={item.label} label={item.label} value={item.value} />
            ))}
          </div>
        </div>
      </div>

      <div className="history-overview-grid">
        {overviewCards.map((item) => (
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
          <LlmRankingPanel items={rankedLlmCards} />
          <HistoryArchivePanel items={llmTrend} />
        </>
      ) : (
        <>
          <PredictorHighlightStrip leaders={predictorLeaders} />
          <PredictorDesk items={rankedPredictorCards} />
        </>
      )}
    </section>
  );
}
