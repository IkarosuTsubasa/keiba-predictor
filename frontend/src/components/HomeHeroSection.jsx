import React, { useMemo } from "react";
import { buildHomeHeroSummary } from "../lib/homepage";

export default function HomeHeroSection({ data, search = "" }) {
  const summary = useMemo(() => buildHomeHeroSummary(data, search), [data, search]);

  return (
    <section className="home-hero" id="home-hero">
      <div className="home-hero__copy">
        <span className="home-section-eyebrow">{summary.eyebrow}</span>
        <h1>{summary.title}</h1>
        <p>{summary.description}</p>

        <div className="home-hero__actions">
          <a className="home-hero__primary" href={summary.primaryCtaHref}>
            {summary.primaryCtaLabel}
          </a>
        </div>
      </div>
    </section>
  );
}
