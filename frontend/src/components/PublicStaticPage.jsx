import React from "react";

export default function PublicStaticPage({ page }) {
  const sections = Array.isArray(page?.sections) ? page.sections : [];

  return (
    <section className="public-static-page">
      <div className="public-static-page__hero">
        <span className="public-static-page__eyebrow">Guide & Policy</span>
        <h1>{page?.title || ""}</h1>
        {page?.lead ? <p>{page.lead}</p> : null}
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
