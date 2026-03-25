import React, { useEffect, useState } from "react";
import FilterBar from "./FilterBar";

function normalizePath(pathname) {
  return String(pathname || "").replace(/\/+$/, "") || "/";
}

function SideNavLink({ href, label, note, active = false, compact = false }) {
  const className = [
    "public-side-nav__link",
    active ? "is-active" : "",
    compact ? "is-compact" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <a className={className} href={href} aria-current={active ? "page" : undefined}>
      <strong>{label}</strong>
      {note ? <span>{note}</span> : null}
    </a>
  );
}

export default function PublicSideNav({
  pathname = "/keiba",
  mode = "home",
  detailHref = "",
  detailTitle = "",
  data,
  search = "",
  onApplyFilters,
  showTargetFilter = false,
}) {
  const [activeHash, setActiveHash] = useState("");
  const normalizedPath = normalizePath(pathname);
  const normalizedDetailHref = detailHref || pathname;

  useEffect(() => {
    const syncHash = () => setActiveHash(window.location.hash || "");
    syncHash();
    window.addEventListener("hashchange", syncHash);
    return () => window.removeEventListener("hashchange", syncHash);
  }, [normalizedPath, normalizedDetailHref]);

  const primaryItems = [
    {
      href: "/keiba",
      label: "予測一覧",
      note: "本日の公開レース",
      active: normalizedPath === "/keiba",
    },
    {
      href: "/keiba/history",
      label: "履歴分析",
      note: "月間・年間・累計",
      active: normalizedPath === "/keiba/history",
    },
  ];

  const detailItems =
    mode === "detail"
      ? [
          {
            href: `${normalizedDetailHref}#race-detail-summary`,
            label: "レース概要",
            note: "基本情報と要点",
            active: !activeHash || activeHash === "#race-detail-summary",
          },
          {
            href: `${normalizedDetailHref}#race-detail-models`,
            label: "買い目比較",
            note: "AIモデル別の買い目",
            active: activeHash === "#race-detail-models",
          },
          {
            href: `${normalizedDetailHref}#race-detail-compare`,
            label: "本命比較",
            note: "量化モデルの本命比較",
            active: activeHash === "#race-detail-compare",
          },
        ]
      : [];

  const currentPageLabel =
    mode === "detail"
      ? detailTitle || "レース詳細"
      : mode === "history"
        ? "履歴分析"
        : mode === "static"
          ? "インフォメーション"
          : "予測一覧";

  const focusText =
    mode === "detail"
      ? "このレースの買い目、比較、結果を一つの流れで確認できます。"
      : mode === "history"
        ? "期間ごとの成績比較と量化モデルの傾向をまとめて確認できます。"
        : mode === "static"
          ? "サイトの考え方や利用上の案内をまとめています。"
          : "公開中のレースとモデル予測を一覧で確認できます。";

  return (
    <aside className="public-side-nav" aria-label="サイトナビゲーション">
      <div className="public-side-nav__panel">
        <div className="public-side-nav__brand">
          <strong className="public-side-nav__title">いかいもAI競馬</strong>
          <p className="public-side-nav__lead">
            予測一覧、履歴分析、単場詳細を同じ流れで見比べられる公開ビューです。
          </p>
        </div>

        <div className="public-side-nav__section">
          <span className="public-side-nav__section-label">ナビゲーション</span>
          <nav className="public-side-nav__links">
            {primaryItems.map((item) => (
              <SideNavLink
                key={item.href}
                href={item.href}
                label={item.label}
                note={item.note}
                active={item.active}
              />
            ))}
          </nav>
        </div>

        {detailItems.length ? (
          <div className="public-side-nav__section">
            <span className="public-side-nav__section-label">単場詳細</span>
            <nav className="public-side-nav__links">
              {detailItems.map((item) => (
                <SideNavLink
                  key={item.href}
                  href={item.href}
                  label={item.label}
                  note={item.note}
                  active={item.active}
                  compact
                />
              ))}
            </nav>
          </div>
        ) : null}

        {showTargetFilter && onApplyFilters ? (
          <div className="public-side-nav__section public-side-nav__section--filter">
            <FilterBar
              data={data}
              search={search}
              onApply={onApplyFilters}
              className="app-filter-bar--sidebar"
            />
          </div>
        ) : null}

        <div className="public-side-nav__focus">
          <span className="public-side-nav__focus-label">現在表示中</span>
          <strong>{currentPageLabel}</strong>
          <p>{focusText}</p>
        </div>
      </div>
    </aside>
  );
}
