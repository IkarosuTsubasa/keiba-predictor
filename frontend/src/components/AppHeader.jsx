import React from "react";
import FilterBar from "./FilterBar";

const HEADER_LINKS = [
  { href: "/keiba", label: "公開レース" },
  { href: "/keiba/history", label: "検証成績" },
  { href: "/keiba/reports", label: "履歴・結果" },
  { href: "/keiba/guide", label: "使い方ガイド" },
];

function normalizePath(pathname) {
  return String(pathname || "").replace(/\/+$/, "") || "/";
}

function isHeaderLinkActive(pathname, href) {
  const current = normalizePath(pathname);
  if (href === "/keiba") {
    return current === "/keiba" || current.startsWith("/keiba/race/");
  }
  if (href === "/keiba/reports") {
    return current === href || current.startsWith("/keiba/reports/");
  }
  return current === href;
}

export default function AppHeader({
  data,
  search,
  onApplyFilters,
  showFilters = true,
  nextPrediction = null,
}) {
  const innerClassName = [
    "app-header__inner",
    showFilters ? "" : "app-header__inner--single",
  ]
    .filter(Boolean)
    .join(" ");
  const brandRowClassName = [
    "app-header__brand-row",
    showFilters ? "" : "app-header__brand-row--single",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <header className="app-header">
      <div className={innerClassName}>
        <div className={brandRowClassName}>
          <div className="app-header__brand">
            <a className="app-header__brand-link" href="/keiba">
              <img
                className="app-header__logo"
                src="/keiba/site-icon.png"
                alt="いかいもAI競馬"
              />
              <div className="app-header__brand-copy">
                <span className="app-header__tag">競馬分析サイト</span>
                <strong className="app-header__title">いかいもAI競馬</strong>
              </div>
            </a>
          </div>

          <nav className="app-header__nav" aria-label="サイトナビゲーション">
            {HEADER_LINKS.map((item) => (
              <a
                key={item.href}
                href={item.href}
                aria-current={isHeaderLinkActive(window.location.pathname, item.href) ? "page" : undefined}
              >
                {item.label}
              </a>
            ))}
          </nav>

          {nextPrediction?.title ? (
            <section
              className="app-header__next-prediction"
              aria-label={nextPrediction.label || "次の予測"}
            >
              <span className="app-header__next-prediction-label">
                {nextPrediction.label || "次の予測"}
              </span>
              <strong className="app-header__next-prediction-title">
                {nextPrediction.title}
              </strong>
              <span className="app-header__next-prediction-time">
                {nextPrediction.publishLabel || "公開時刻調整中"}
              </span>
            </section>
          ) : null}
        </div>

        {showFilters ? (
          <div className="app-header__aside">
            <div className="app-header__filters">
              <FilterBar data={data} search={search} onApply={onApplyFilters} />
            </div>
          </div>
        ) : null}
      </div>
    </header>
  );
}
