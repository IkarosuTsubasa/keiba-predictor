import React from "react";
import { BEGINNER_GUIDE_LINKS, BEGINNER_GUIDE_SECTION } from "../lib/siteCopy";

export default function BeginnerGuideSection() {
  return (
    <section className="home-section-card home-section-card--compact" id="home-guide-nav">
      <div className="home-section-head">
        <span className="home-section-eyebrow">{BEGINNER_GUIDE_SECTION.eyebrow}</span>
        <h2>{BEGINNER_GUIDE_SECTION.title}</h2>
        <p>{BEGINNER_GUIDE_SECTION.description}</p>
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
