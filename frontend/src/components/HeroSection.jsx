import React, { useMemo } from "react";
import AiQuickPickCard from "./AiQuickPickCard";
import HeroRacePanel from "./HeroRacePanel";

function sortCards(cards) {
  return [...(cards || [])].sort((a, b) => {
    const aValue = Number.parseFloat(String(a?.roi_text || "").replace("%", ""));
    const bValue = Number.parseFloat(String(b?.roi_text || "").replace("%", ""));
    const safeA = Number.isFinite(aValue) ? aValue : -999;
    const safeB = Number.isFinite(bValue) ? bValue : -999;
    return safeB - safeA;
  });
}

export default function HeroSection({ data }) {
  const leadRace = data?.hero?.lead_race || {};
  const leader = data?.hero?.leader || {};
  const quickCards = useMemo(() => sortCards(leadRace?.cards || []), [leadRace]);

  return (
    <section className="hero-section">
      <HeroRacePanel race={leadRace} leader={leader} />
      <div className="hero-section__sidebar">
        <div className="hero-section__sidebar-head">
          <span className="section-kicker">Quick Picks</span>
          <h2>4 AI の即時予想</h2>
        </div>
        <div className="hero-section__quick-grid">
          {quickCards.map((card) => (
            <AiQuickPickCard key={`${leadRace?.run_id}-${card.engine}`} card={card} />
          ))}
        </div>
      </div>
    </section>
  );
}
