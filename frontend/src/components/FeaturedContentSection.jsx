import React from "react";
import { FEATURED_CONTENT_ITEMS, FEATURED_CONTENT_SECTION } from "../lib/siteCopy";

export default function FeaturedContentSection() {
  return (
    <section className="home-section-card" id="home-featured">
      <div className="home-section-head">
        <span className="home-section-eyebrow">{FEATURED_CONTENT_SECTION.eyebrow}</span>
        <h2>{FEATURED_CONTENT_SECTION.title}</h2>
        <p>{FEATURED_CONTENT_SECTION.description}</p>
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
            <a href={item.href}>{item.cta}</a>
          </article>
        ))}
      </div>
    </section>
  );
}
