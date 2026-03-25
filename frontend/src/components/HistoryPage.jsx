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

function TabButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      className={active ? "is-active" : ""}
      onClick={onClick}
    >
      {children}
    </button>
  );
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
    return null;
  }

  return (
    <article className="history-hero-leader">
      <div className="history-hero-leader__top">
        <span className="history-hero-leader__eyebrow">最高回収</span>
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

function LlmRankBoard({ items }) {
  const ranked = sortByPercent(items, "roi_text");
  const leader = ranked[0] || null;
  const rest = ranked.slice(1);

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
          {ranked.map((item, index) => (
            <article key={item.engine} className="history-rank-item">
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
          ))}
        </div>
      </div>

      {rest.length ? (
        <div className="history-compact-grid">
          {rest.map((item) => (
            <article key={`compact-${item.engine}`} className="history-compact-card">
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
          <h2>直近の推移と日次比較</h2>
        </div>
      </div>

      <div className="history-trend-table">
        {items.map((item) => (
          <article key={item.date} className="history-trend-row">
            <div className="history-trend-row__summary">
              <span>{item.date || "-"}</span>
              <strong>{item.roi_text || "-"}</strong>
              <em>{`損益 ${formatYen(item.profit_yen || 0)}`}</em>
            </div>
            <div className="history-trend-row__chips">
              {(item.cards || []).map((card) => (
                <span key={`${item.date}-${card.engine}`}>
                  <b>{card.label || "-"}</b>
                  <strong>{card.roi_text || "-"}</strong>
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function PredictorMatrix({ items }) {
  const rankedTop1 = sortByPercent(items, "top1_hit_rate_text");

  return (
    <section className="history-panel history-panel--predictor">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">比較表</span>
          <h2>量化モデル比較テーブル</h2>
        </div>
      </div>

      <div className="history-predictor-table">
        <div className="history-predictor-table__head">
          <span>モデル</span>
          <span>Top1</span>
          <span>Top1複勝圏</span>
          <span>Top3</span>
          <span>Top3完全</span>
          <span>Top5→Top3</span>
          <span>Samples</span>
        </div>
        {rankedTop1.map((item, index) => (
          <article key={item.predictor_id} className="history-predictor-table__row">
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
    </section>
  );
}

function PredictorHighlightStrip({ items }) {
  const top1 = sortByPercent(items, "top1_hit_rate_text")[0] || null;
  const top3 = sortByPercent(items, "top3_hit_rate_text")[0] || null;
  const exact = sortByPercent(items, "top3_exact_rate_text")[0] || null;

  return (
    <div className="history-highlight-strip">
      <OverviewCard
        label="Top1 最良"
        value={top1?.label || "-"}
        note={top1?.top1_hit_rate_text || "-"}
        accent
      />
      <OverviewCard
        label="Top3 最良"
        value={top3?.label || "-"}
        note={top3?.top3_hit_rate_text || "-"}
      />
      <OverviewCard
        label="完全一致 最良"
        value={exact?.label || "-"}
        note={exact?.top3_exact_rate_text || "-"}
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

  const llmLeader = useMemo(
    () => sortByPercent(llmCards, "roi_text")[0] || null,
    [llmCards],
  );

  const llmOverview = [
    {
      label: "期間回収率",
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
      label: "総サンプル数",
      value: String(predictorPeriod?.totals?.samples || 0),
      note: `${activePeriodLabel}の累積実績`,
      accent: true,
    },
    {
      label: "対象モデル数",
      value: String(predictorCards.length),
      note: "Predictor V1-V5",
      accent: false,
    },
    {
      label: "比較粒度",
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
          <h1>期間を切り替えて、モデルの強さを見る</h1>
          <p>
            日次の断片ではなく、30日、365日、累計のスパンで回収率と命中傾向を整理した公開ヒストリーページです。
          </p>
        </div>

        <div className="history-hero__controls">
          <div className="history-hero__tabs" role="tablist" aria-label="履歴区分">
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
          <LlmRankBoard items={llmCards} />
          <LlmTrendTable items={llmTrend} />
        </>
      ) : (
        <>
          <PredictorHighlightStrip items={predictorCards} />
          <PredictorMatrix items={predictorCards} />
        </>
      )}
    </section>
  );
}
