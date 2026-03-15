import React from "react";

export default function EmptyRaceState() {
  return (
    <section className="empty-race-state">
      <span className="empty-race-state__eyebrow">No Public Races</span>
      <h2>公開中のレースはまだありません</h2>
      <p>日付を切り替えると、ほかの公開レースが表示される場合があります。</p>
    </section>
  );
}
