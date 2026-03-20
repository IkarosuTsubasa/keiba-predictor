import React from "react";
import FilterBar from "./FilterBar";

export default function AppHeader({ data, search, onApplyFilters, showFilters = true }) {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <a className="app-header__brand-link" href="/keiba">
          <img className="app-header__logo" src="/keiba/site-icon.png" alt="いかいもAI競馬" />
          <div className="app-header__brand-copy">
            <span className="app-header__tag">Racing Intelligence</span>
            <strong className="app-header__title">いかいもAI競馬</strong>
          </div>
        </a>
      </div>

      {showFilters ? (
        <div className="app-header__filters">
          <FilterBar data={data} search={search} onApply={onApplyFilters} />
        </div>
      ) : null}
    </header>
  );
}
