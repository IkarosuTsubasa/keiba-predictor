import React, { useMemo, useState } from "react";

const PREDICTOR_LABELS = {
  top1: "本命1着率",
  top1InTop3: "本命複勝圏率",
  top3: "予測上位3頭馬券内率",
  top3Exact: "予測上位3頭完全的中率",
  top5to3: "予測上位5頭馬券内カバー率",
};

const PREDICTOR_SORT_OPTIONS = [
  { key: "top1_hit_rate_text", label: PREDICTOR_LABELS.top1 },
  { key: "top1_in_top3_rate_text", label: PREDICTOR_LABELS.top1InTop3 },
  { key: "top3_hit_rate_text", label: PREDICTOR_LABELS.top3 },
  { key: "top3_exact_rate_text", label: PREDICTOR_LABELS.top3Exact },
  { key: "top5_to_top3_hit_rate_text", label: PREDICTOR_LABELS.top5to3 },
];

const PREDICTOR_PROFILES = {
  main: {
    lead: "基礎能力、近走内容、距離と馬場の適性を広く確認する標準型です。",
    tags: ["総合型", "近走", "距離・馬場"],
  },
  v2_opus: {
    lead: "能力指数に加えて展開、斤量、オッズのバランスを見る精度重視型です。",
    tags: ["能力", "展開", "オッズ"],
  },
  v3_premium: {
    lead: "コース条件、脚質適性、市場評価を深く織り込む高精度型です。",
    tags: ["条件適性", "脚質", "市場評価"],
  },
  v4_gemini: {
    lead: "レース文脈、騎手相性、馬場速度への適応を強く見る文脈重視型です。",
    tags: ["文脈適性", "騎手相性", "馬場速度"],
  },
  v5_stacking: {
    lead: "複数モデルと複数オッズを束ねて判断する統合アンサンブル型です。",
    tags: ["統合型", "スタッキング", "複数オッズ"],
  },
  v6_kiwami: {
    lead: "市場事前分布と能力残差、レース内順位学習を重ねる高精度モデルです。",
    tags: ["高精度型", "市場融合", "順位学習"],
  },
};

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

function sortPredictorRows(items, key, direction = "desc") {
  const multiplier = direction === "asc" ? 1 : -1;
  return [...(items || [])].sort((left, right) => {
    const leftValue = parsePercentText(left?.[key]) ?? -9999;
    const rightValue = parsePercentText(right?.[key]) ?? -9999;
    if (leftValue !== rightValue) {
      return (leftValue - rightValue) * multiplier;
    }
    return String(left?.label || "").localeCompare(String(right?.label || ""), "ja");
  });
}

function rankLabel(index) {
  return `No.${String(index + 1).padStart(2, "0")}`;
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

function LlmLeaderCard({ leader }) {
  if (!leader) {
    return (
      <article className="history-hero-leader">
        <div className="history-hero-leader__body">
          <strong>-</strong>
          <p>この期間の公開データはまだありません。</p>
        </div>
      </article>
    );
  }

  return (
    <article className="history-hero-leader">
      <div className="history-hero-leader__top">
        <span className="history-hero-leader__eyebrow">最高回収</span>
        <span className="history-hero-leader__badge">{leader.label || "-"}</span>
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
          <span className="history-panel__eyebrow">回収比較</span>
          <h2>LLM 回収ランキング</h2>
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
            <EmptyState>この期間のランキングデータはまだありません。</EmptyState>
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
          <span className="history-panel__eyebrow">推移</span>
          <h2>日別アーカイブ</h2>
        </div>
      </div>

      {items.length ? (
        <div className="history-archive-table">
          <div className="history-archive-table__head">
            <span>対象日</span>
            <span>首位モデル</span>
            <span>公開数</span>
            <span>損益</span>
            <span>ROI</span>
          </div>
          {items.map((item) => {
            const leader = pickDailyLeader(item.cards || []);
            return (
              <article key={item.date} className="history-archive-table__row">
                <span>{item.date || "-"}</span>
                <div className="history-archive-table__model">
                  <strong>{leader?.label || "集計なし"}</strong>
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
        label={`${PREDICTOR_LABELS.top1}トップ`}
        value={leaders.top1?.label || "-"}
        note={leaders.top1?.top1_hit_rate_text || "-"}
        accent
      />
      <OverviewCard
        label={`${PREDICTOR_LABELS.top3}トップ`}
        value={leaders.top3?.label || "-"}
        note={leaders.top3?.top3_hit_rate_text || "-"}
      />
      <OverviewCard
        label={`${PREDICTOR_LABELS.top5to3}トップ`}
        value={leaders.top5to3?.label || "-"}
        note={leaders.top5to3?.top5_to_top3_hit_rate_text || "-"}
      />
    </div>
  );
}

function PredictorSortButton({ active, direction, onClick, children }) {
  return (
    <button
      type="button"
      className={`history-predictor-sort${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      <span>{children}</span>
      <em>{active ? (direction === "asc" ? "↑" : "↓") : "↕"}</em>
    </button>
  );
}

function PredictorDesk({ items }) {
  const [sortKey, setSortKey] = useState("top1_hit_rate_text");
  const [sortDirection, setSortDirection] = useState("desc");

  const sortedItems = useMemo(
    () => sortPredictorRows(items, sortKey, sortDirection),
    [items, sortDirection, sortKey],
  );

  const handleSort = (nextKey) => {
    if (nextKey === sortKey) {
      setSortDirection((current) => (current === "desc" ? "asc" : "desc"));
      return;
    }
    setSortKey(nextKey);
    setSortDirection("desc");
  };

  return (
    <section className="history-panel history-panel--predictor">
      <div className="history-panel__head">
        <div>
          <span className="history-panel__eyebrow">量化比較</span>
          <h2>定量モデル比較</h2>
        </div>
      </div>

      {sortedItems.length ? (
        <>
          <div className="history-predictor-table">
            <div className="history-predictor-table__head">
              <span>モデル</span>
              {PREDICTOR_SORT_OPTIONS.map((option) => (
                <PredictorSortButton
                  key={option.key}
                  active={sortKey === option.key}
                  direction={sortDirection}
                  onClick={() => handleSort(option.key)}
                >
                  {option.label}
                </PredictorSortButton>
              ))}
            </div>
            {sortedItems.map((item, index) => (
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
              </article>
            ))}
          </div>

          <div className="history-predictor-card-grid">
            {items.map((item) => {
              const profile = PREDICTOR_PROFILES[item.predictor_id] || {
                lead: "能力、適性、オッズを横断して評価する定量モデルです。",
                tags: ["定量モデル"],
              };
              return (
                <article
                  key={`predictor-card-${item.predictor_id || item.label}`}
                  className="history-predictor-card"
                >
                  <h3>{item.label || "-"}</h3>
                  <p className="history-predictor-card__lead">{profile.lead}</p>
                  <div className="history-predictor-card__tags">
                    {profile.tags.map((tag) => (
                      <span
                        key={`${item.predictor_id}-${tag}`}
                        className="history-predictor-card__tag"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </article>
              );
            })}
          </div>
        </>
      ) : (
        <EmptyState>定量モデルの履歴データはまだありません。</EmptyState>
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
    { key: "days_30", label: "月間" },
    { key: "days_365", label: "年間" },
    { key: "all_time", label: "累計" },
  ];

  const activePeriodLabel =
    periodTabs.find((item) => item.key === periodKey)?.label || "月間";

  const llmPeriod = llmPeriods?.[periodKey] || { cards: [], totals: {}, trend: [] };
  const predictorPeriod = predictorPeriods?.[periodKey] || { cards: [] };

  const llmCards = Array.isArray(llmPeriod.cards) ? llmPeriod.cards : [];
  const llmTrend = Array.isArray(llmPeriod.trend) ? llmPeriod.trend : [];
  const predictorCards = Array.isArray(predictorPeriod.cards) ? predictorPeriod.cards : [];

  const rankedLlmCards = useMemo(() => sortByPercent(llmCards, "roi_text"), [llmCards]);
  const predictorLeaders = useMemo(
    () => ({
      top1: sortByPercent(predictorCards, "top1_hit_rate_text")[0] || null,
      top3: sortByPercent(predictorCards, "top3_hit_rate_text")[0] || null,
      top5to3:
        sortByPercent(predictorCards, "top5_to_top3_hit_rate_text")[0] || null,
    }),
    [predictorCards],
  );

  const overviewCards =
    groupKey === "llm"
      ? [
          {
            label: "総合ROI",
            value: llmPeriod?.totals?.roi_text || "-",
            note: `損益 ${formatYen(llmPeriod?.totals?.profit_yen || 0)}`,
            accent: true,
          },
          {
            label: "最高回収",
            value: rankedLlmCards[0]?.label || "-",
            note: rankedLlmCards[0]?.roi_text || "-",
          },
          {
            label: "対象レース",
            value: `${llmPeriod?.totals?.runs || 0}レース`,
            note: `${activePeriodLabel}の公開結果`,
          },
        ]
      : [];

  return (
    <section className="history-page">
      <div className="history-hero">
        <div className="history-hero__copy">
          <span className="history-hero__eyebrow">履歴分析</span>
          <h1>成績比較</h1>
          <p>
            AI モデルと定量モデルの成績を、月間・年間・累計で確認できます。
          </p>
        </div>

        <div className="history-hero__controls">
          <div className="history-hero__tabs" role="tablist" aria-label="履歴グループ">
            <TabButton active={groupKey === "llm"} onClick={() => setGroupKey("llm")}>
              AIモデル
            </TabButton>
            <TabButton
              active={groupKey === "predictor"}
              onClick={() => setGroupKey("predictor")}
            >
              定量モデル
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

      {overviewCards.length ? (
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
      ) : null}

      {groupKey === "llm" ? (
        <>
          <LlmRankingPanel items={rankedLlmCards} />
          <HistoryArchivePanel items={llmTrend} />
        </>
      ) : (
        <>
          <PredictorHighlightStrip leaders={predictorLeaders} />
          <PredictorDesk items={predictorCards} />
        </>
      )}
    </section>
  );
}

