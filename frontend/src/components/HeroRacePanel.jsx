import React from "react";
import CountdownBadge from "./CountdownBadge";
import ModelBadge from "./ModelBadge";

function parseTopMark(card) {
  const text = String(card?.marks_text || "");
  const match = text.match(/([◎○▲△☆])\s*([0-9]+)/);
  return match ? `${match[1]}${match[2]}` : "未生成";
}

export default function HeroRacePanel({ race, leader }) {
  const badges = [
    race?.location,
    race?.distance_label,
    race?.track_condition ? `馬場 ${race.track_condition}` : "",
  ].filter(Boolean);

  return (
    <section className="hero-race-panel">
      <div className="hero-race-panel__topline">
        <span className="section-kicker">Today&apos;s Main Race</span>
        <CountdownBadge targetText={race?.scheduled_off_time} />
      </div>
      <div className="hero-race-panel__main">
        <div className="hero-race-panel__copy">
          <h1>AI競馬予想バトル</h1>
          <p>4つのAIが同じレースを予想。買い目と回収率を毎日公開。</p>
          <div className="hero-race-panel__support">レース前に予想、レース後に結果を反映</div>
        </div>
        <div className="hero-race-panel__focus">
          <span className="hero-race-panel__focus-label">今日主赛</span>
          <strong className="hero-race-panel__focus-race">{race?.race_title || "公開レース準備中"}</strong>
          <div className="hero-race-panel__badges">
            {badges.map((badge) => (
              <span key={badge} className="hero-race-panel__badge">
                {badge}
              </span>
            ))}
          </div>
          <p className="hero-race-panel__result">{race?.actual_text || "結果未確定"}</p>
        </div>
      </div>
      <div className="hero-race-panel__footer">
        <div className="hero-race-panel__leader">
          <span className="hero-race-panel__leader-label">現在领先模型</span>
          <div className="hero-race-panel__leader-row">
            <ModelBadge engine={leader?.engine} label={leader?.label || "N/A"} />
            <strong>{leader?.roi_text || "-"}</strong>
          </div>
        </div>
        <div className="hero-race-panel__quickmarks">
          {(race?.cards || []).slice(0, 4).map((card) => (
            <div key={`${race?.run_id}-${card.engine}`} className="hero-race-panel__quickmark">
              <span>{card.label}</span>
              <strong>{parseTopMark(card)}</strong>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
