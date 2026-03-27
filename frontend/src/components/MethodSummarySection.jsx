import React from "react";
import { METHOD_SUMMARY_SECTION, METHOD_SUMMARY_STEPS } from "../lib/siteCopy";

export default function MethodSummarySection() {
  return (
    <section className="home-section-card" id="home-method">
      <div className="home-section-head">
        <span className="home-section-eyebrow">{METHOD_SUMMARY_SECTION.eyebrow}</span>
        <h2>{METHOD_SUMMARY_SECTION.title}</h2>
        <p>{METHOD_SUMMARY_SECTION.description}</p>
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
        <a href="/keiba/methodology">{METHOD_SUMMARY_SECTION.primary_link_label}</a>
        <a href="/keiba/guide">{METHOD_SUMMARY_SECTION.secondary_link_label}</a>
      </div>

      <p className="home-section-note">{METHOD_SUMMARY_SECTION.note}</p>
    </section>
  );
}
