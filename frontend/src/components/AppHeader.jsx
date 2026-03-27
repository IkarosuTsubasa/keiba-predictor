import React from "react";
import FilterBar from "./FilterBar";

const HEADER_LINKS = [
  { href: "/keiba", label: "トップ" },
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
                <span className="app-header__tag">競馬分析サイト</span>
                <strong className="app-header__title">いかいもAI競馬</strong>
              </div>
            </a>
          </div>

          <nav className="app-header__nav" aria-label="サイトナビゲーション">
            {HEADER_LINKS.map((item) => (
              <a key={item.href} href={item.href}>
                {item.label}
              </a>
            ))}
          </nav>
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
