import React from "react";
import FilterBar from "./FilterBar";

export default function AppHeader({ data, search, onApplyFilters }) {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <img className="app-header__logo" src="/keiba/site-icon.png" alt="いかいもAI競馬" />
        <strong className="app-header__title">いかいもAI競馬</strong>
      </div>
      <div className="app-header__filters">
        <FilterBar data={data} search={search} onApply={onApplyFilters} compact />
      </div>
    </header>
  );
}
