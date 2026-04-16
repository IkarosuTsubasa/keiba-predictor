import React from "react";

function predictorLeader(cards, key) {
  const rows = Array.isArray(cards) ? cards : [];
  return [...rows].sort((left, right) => {
    const leftValue = Number(left?.[key] || 0);
    const rightValue = Number(right?.[key] || 0);
    if (leftValue !== rightValue) return rightValue - leftValue;
    return String(left?.label || "").localeCompare(String(right?.label || ""), "ja");
  })[0] || null;
}

function buildRows(cards) {
  const top1 = predictorLeader(cards, "top1_hit_rate");
  const top3 = predictorLeader(cards, "top3_hit_rate");
  const top5 = predictorLeader(cards, "top5_to_top3_hit_rate");

  return [
    {
      title: "本命1着率",
      items: [
        { key: "top1-label", label: top1?.label || "-", value: top1?.top1_hit_rate_text || "-", accent: true },
        { key: "top1-samples", label: "対象", value: `${top1?.samples || 0}レース`, accent: false },
      ],
    },
    {
      title: "上位3頭馬券内率",
      items: [
        { key: "top3-label", label: top3?.label || "-", value: top3?.top3_hit_rate_text || "-", accent: true },
        { key: "top3-samples", label: "対象", value: `${top3?.samples || 0}レース`, accent: false },
      ],
    },
    {
      title: "上位5頭カバー率",
      items: [
        { key: "top5-label", label: top5?.label || "-", value: top5?.top5_to_top3_hit_rate_text || "-", accent: true },
        { key: "top5-samples", label: "対象", value: `${top5?.samples || 0}レース`, accent: false },
      ],
    },
  ];
}

function MetricRow({ title, items }) {
  return (
    <div className="secondary-stats-panel__row">
      <span className="secondary-stats-panel__row-title">{title}</span>
      <div className="secondary-stats-panel__tokens">
        {items.map((item) => (
          <div
            key={item.key}
            className={`secondary-stats-panel__token${item.accent ? " secondary-stats-panel__token--accent" : ""}`}
          >
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SecondaryStatsPanel({ data }) {
  const predictorCards = Array.isArray(data?.daily_predictor?.cards)
    ? data.daily_predictor.cards
    : [];
  const rows = buildRows(predictorCards);

  return (
    <section className="secondary-stats-panel" aria-label="定量モデル指標">
      <div className="secondary-stats-panel__intro">
        <span className="home-section-eyebrow">定量モデル指標</span>
        <h2>6モデルの成績比較</h2>
        <p>回収率ではなく、まずは本命精度と上位候補のカバー率から定量モデルの強さを確認できます。</p>
      </div>
      <div className="secondary-stats-panel__compact">
        {rows.map((row) => (
          <MetricRow key={row.title} title={row.title} items={row.items} />
        ))}
      </div>
    </section>
  );
}
