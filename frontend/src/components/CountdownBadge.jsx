import React, { useEffect, useMemo, useState } from "react";

function buildCountdown(targetText) {
  const target = targetText ? new Date(targetText) : null;
  if (!target || Number.isNaN(target.getTime())) {
    return "COUNTDOWN --:--:--";
  }
  const diff = target.getTime() - Date.now();
  if (diff <= 0) {
    return "STARTED";
  }
  const totalSeconds = Math.floor(diff / 1000);
  const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `COUNTDOWN ${hours}:${minutes}:${seconds}`;
}

export default function CountdownBadge({ targetText }) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!targetText) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [targetText]);

  const text = useMemo(() => buildCountdown(targetText, now), [targetText, now]);
  return <span className="countdown-badge">{text}</span>;
}
