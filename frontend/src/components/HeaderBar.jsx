import React from "react";

export default function HeaderBar({ data, children }) {
  return (
    <header className="header-bar">
      <div className="header-bar__brand">
        <span className="header-bar__tag">AI競馬予想バトル</span>
        <strong className="header-bar__title">いかいも競馬AI</strong>
        <span className="header-bar__subtle">最終更新 {data?.generated_at_label || "-"}</span>
      </div>
      <div className="header-bar__controls">{children}</div>
    </header>
  );
}
