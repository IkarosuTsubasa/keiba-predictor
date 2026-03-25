import React from "react";
import FilterBar from "./FilterBar";

const HEADER_LINKS = [
  { href: "/keiba", label: "ホーム" },
  { href: "/keiba/history", label: "履歴分析" },
  { href: "/keiba/about", label: "このサイトについて" },
  { href: "/keiba/guide", label: "ガイド" },
  { href: "/keiba/methodology", label: "分析方針" },
];

export default function AppHeader({
  data,
  search,
  onApplyFilters,
  showFilters = true,
}) {
  const statusDate = data?.target_date_label || data?.target_date || "公開中";

  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand-row">
          <div className="app-header__brand">
            <a className="app-header__brand-link" href="/keiba">
              <img
                className="app-header__logo"
                src="/keiba/site-icon.png"
                alt="いかいもAI競馬"
              />
              <div className="app-header__brand-copy">
                <span className="app-header__tag">公開競馬分析</span>
                <strong className="app-header__title">いかいもAI競馬</strong>
              </div>
            </a>
          </div>

          <div className="app-header__nav-shell">
            <nav className="app-header__nav" aria-label="公開ナビゲーション">
              {HEADER_LINKS.map((item) => (
                <a key={item.href} href={item.href}>
                  {item.label}
                </a>
              ))}
            </nav>
            <div className="app-header__status" aria-label="公開状況">
              <span className="app-header__status-label">公開日</span>
              <strong className="app-header__status-value">{statusDate}</strong>
            </div>
          </div>
        </div>

        {showFilters ? (
          <div className="app-header__filters">
            <FilterBar data={data} search={search} onApply={onApplyFilters} />
          </div>
        ) : null}
      </div>
    </header>
  );
}
