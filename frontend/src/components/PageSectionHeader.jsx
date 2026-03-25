import React from "react";

function normalizeMetaItem(item) {
  if (item && typeof item === "object") {
    return {
      key: item.key || `${item.label || ""}-${item.value || ""}`,
      label: item.label || "",
      value: item.value || "",
    };
  }
  return {
    key: String(item),
    label: "",
    value: String(item || ""),
  };
}

export default function PageSectionHeader({
  kicker,
  title,
  subtitle = "",
  meta = [],
}) {
  const visibleMeta = (meta || [])
    .filter(Boolean)
    .map(normalizeMetaItem)
    .filter((item) => item.value);

  return (
    <div className="page-section-header">
      <div className="page-section-header__copy">
        <span className="page-section-header__kicker">{kicker}</span>
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {visibleMeta.length ? (
        <div className="page-section-header__meta" aria-label={kicker || title}>
          {visibleMeta.map((item) => (
            <article key={item.key} className="page-section-header__meta-card">
              {item.label ? (
                <span className="page-section-header__meta-label">{item.label}</span>
              ) : null}
              <strong className="page-section-header__meta-value">{item.value}</strong>
            </article>
          ))}
        </div>
      ) : null}
    </div>
  );
}
