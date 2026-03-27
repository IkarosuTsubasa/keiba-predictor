import React from "react";
import { FEATURED_CONTENT_ITEMS } from "../lib/homepage";

export default function FeaturedContentSection() {
  return (
    <section className="home-section-card" id="home-featured">
      <div className="home-section-head">
        <span className="home-section-eyebrow">Featured Content</span>
        <h2>最新の深掘り分析</h2>
        <p>
          分析方法、読み方、履歴検証の3方向から、公開レースをどう読むかを整理した内容をまとめています。
        </p>
      </div>

      <div className="home-featured-grid">
        {FEATURED_CONTENT_ITEMS.map((item) => (
          <article key={item.id} className="home-featured-card">
            <div className="home-featured-card__top">
              <span className="home-featured-card__category">{item.category}</span>
            </div>
            <h3>{item.title}</h3>
            <p>{item.excerpt}</p>
            <div className="home-featured-card__tags">
              {item.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <a href={item.href}>続きを読む</a>
          </article>
        ))}
      </div>
    </section>
  );
}
