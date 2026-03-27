import React from "react";
import { BEGINNER_GUIDE_LINKS } from "../lib/homepage";

export default function BeginnerGuideSection() {
  return (
    <section className="home-section-card home-section-card--compact" id="home-guide-nav">
      <div className="home-section-head">
        <span className="home-section-eyebrow">Guide</span>
        <h2>初めて読む人への入口</h2>
        <p>
          いきなり買い目一覧に入るのではなく、サイトの前提、見方、分析方法、履歴検証の順に読むと全体像を把握しやすくなります。
        </p>
      </div>

      <div className="home-beginner-grid">
        {BEGINNER_GUIDE_LINKS.map((item) => (
          <a key={item.href} className="home-beginner-card" href={item.href}>
            <strong>{item.title}</strong>
            <span>{item.note}</span>
          </a>
        ))}
      </div>
    </section>
  );
}
