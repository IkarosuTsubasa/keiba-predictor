import React from "react";

export default function PageSectionHeader({ kicker, title, subtitle = "", meta = [] }) {
  const visibleMeta = (meta || []).filter(Boolean);

  return (
    <div className="page-section-header">
      <div className="page-section-header__copy">
        <span className="page-section-header__kicker">{kicker}</span>
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {visibleMeta.length ? (
        <div className="page-section-header__meta">
          {visibleMeta.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
