import React from "react";
import CountdownBadge from "./CountdownBadge";

export default function RaceCommandCard({ race }) {
  const items = [
    race?.location,
    race?.scope_label,
    race?.distance_label,
    race?.track_condition ? `馬場 ${race.track_condition}` : "",
  ].filter(Boolean);

  return (
    <div className="race-command-card">
      <div className="race-command-card__chips">
        {items.map((item) => (
          <span key={item} className="race-command-card__chip">
            {item}
          </span>
        ))}
      </div>
      <CountdownBadge targetText={race?.scheduled_off_time} />
    </div>
  );
}
