import React from "react";
import { METHOD_SUMMARY_STEPS } from "../lib/homepage";

export default function MethodSummarySection() {
  return (
    <section className="home-section-card" id="home-method">
      <div className="home-section-head">
        <span className="home-section-eyebrow">Method</span>
        <h2>予想の作り方</h2>
        <p>
          定量モデルで有力馬を抽出し、LLMで買い目構成を比較し、その後の結果と履歴まで同じ流れで検証します。
        </p>
      </div>

      <div className="home-method-grid">
        {METHOD_SUMMARY_STEPS.map((item) => (
          <article key={item.step} className="home-method-card">
            <span className="home-method-card__step">{item.step}</span>
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </article>
        ))}
      </div>

      <div className="home-section-links">
        <a href="/keiba/methodology">分析方針を詳しく見る</a>
        <a href="/keiba/guide">読み方ガイドを見る</a>
      </div>

      <p className="home-section-note">
        最終的な判断と馬券購入は利用者自身の責任で行ってください。
      </p>
    </section>
  );
}
