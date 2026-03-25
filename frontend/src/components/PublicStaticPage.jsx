import React from "react";

export default function PublicStaticPage({ page }) {
  const sections = Array.isArray(page?.sections) ? page.sections : [];
  const meta = Array.isArray(page?.meta) ? page.meta : [];

  return (
    <section className="public-static-page">
      <div className="public-static-page__hero">
        <div className="public-static-page__hero-copy">
          <span className="public-static-page__eyebrow">{page?.kicker || "ガイド・ポリシー"}</span>
          <h1>{page?.title || ""}</h1>
          {page?.lead ? <p>{page.lead}</p> : null}
        </div>
        {meta.length ? (
          <div className="public-static-page__meta" aria-label="ページ要点">
            {meta.map((item) => (
              <article key={item.label} className="public-static-page__meta-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </article>
            ))}
          </div>
        ) : null}
      </div>

      <div className="public-static-page__body">
        {sections.map((section) => (
          <article key={section.heading} className="public-static-page__section">
            <h2>{section.heading}</h2>
            {(section.paragraphs || []).map((text) => (
              <p key={text}>{text}</p>
            ))}
            {(section.bullets || []).length ? (
              <ul>
                {section.bullets.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
