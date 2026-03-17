import React from "react";

function safeRoiText(value) {
  const text = String(value || "").trim();
  return text || "-";
}

function buildRowItems(totalLabel, totalRoi, cards) {
  const items = [{ key: "total", label: totalLabel, value: safeRoiText(totalRoi), accent: true }];
  for (const card of cards || []) {
    items.push({
      key: card.engine || card.label,
      label: card.label || "-",
      value: safeRoiText(card.roi_text),
      accent: false,
    });
  }
  return items;
}

function RoiRow({ title, items }) {
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
  const allTimeItems = buildRowItems("全体", data?.all_time_roi?.totals?.roi_text, data?.all_time_roi?.cards || []);
  const dailyItems = buildRowItems("全体", data?.totals?.roi_text, data?.summary_cards || []);

  return (
    <section className="secondary-stats-panel" aria-label="回収率サマリー">
      <div className="secondary-stats-panel__compact">
        <RoiRow title="通算回収率" items={allTimeItems} />
        <RoiRow title="当日回収率" items={dailyItems} />
      </div>
    </section>
  );
}
