import React from "react";

export default function EmptyRaceState({ agentMode = false }) {
  return (
    <section className="empty-race-state">
      <span className="empty-race-state__eyebrow">公開レース</span>
      <h2>{agentMode ? "公開中のAI予測はありません" : "公開中のレースはありません"}</h2>
      <p>
        {agentMode
          ? "Consoleで登録したレースのAI予測が生成されると、ここにレース一覧が表示されます。"
          : "日付を切り替えると、ほかの公開レースが表示される場合があります。"}
      </p>
    </section>
  );
}
