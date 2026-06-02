import React, { useEffect, useState } from "react";
import FilterBar from "./FilterBar";

function normalizePath(pathname) {
  return String(pathname || "").replace(/\/+$/, "") || "/";
}

function SideNavLink({ href, label, note, active = false, compact = false }) {
  const className = ["public-side-nav__link", active ? "is-active" : "", compact ? "is-compact" : ""]
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
  agentMode = false,
  search = "",
  onApplyFilters,
  showTargetFilter = false,
}) {
  const [activeHash, setActiveHash] = useState("");
  const normalizedPath = normalizePath(pathname);
  const normalizedDetailHref = detailHref || pathname;
  const shouldShowTargetFilter = Boolean(onApplyFilters) && (showTargetFilter || mode !== "static");
  const isAgentMode =
    Boolean(agentMode) ||
    String(data?.race?.source_type || "").trim() === "agent_prediction" ||
    Boolean(data?.race?.agent_prediction) ||
    (Array.isArray(data?.races) &&
      data.races.some((race) => String(race?.source_type || "").trim() === "agent_prediction"));

  useEffect(() => {
    const syncHash = () => setActiveHash(window.location.hash || "");
    syncHash();
    window.addEventListener("hashchange", syncHash);
    return () => window.removeEventListener("hashchange", syncHash);
  }, [normalizedPath, normalizedDetailHref]);

  const primaryItems = [
    {
      href: "/keiba",
      label: "トップページ",
      note: isAgentMode ? "公開レース一覧" : "公開レースと導読",
      active: normalizedPath === "/keiba",
    },
    {
      href: "/keiba/history",
      label: "履歴分析",
      note: isAgentMode ? "予測命中率の分析" : "月間・年間・累計の比較",
      active: normalizedPath === "/keiba/history",
    },
    {
      href: "/keiba/reports",
      label: "私の日報",
      note: "日次の振り返りアーカイブ",
      active: normalizedPath === "/keiba/reports" || normalizedPath.startsWith("/keiba/reports/"),
    },
  ].filter(Boolean);

  const detailItems =
    mode === "detail"
      ? isAgentMode
        ? [
          {
            href: `${normalizedDetailHref}#race-detail-summary`,
            label: "レース概要",
            note: "基本情報と結果",
            active: !activeHash || activeHash === "#race-detail-summary",
          },
          {
            href: `${normalizedDetailHref}#race-detail-agent`,
            label: "予測メモ",
            note: "判断と買い目候補",
            active: activeHash === "#race-detail-agent",
          },
          {
            href: `${normalizedDetailHref}#race-detail-agent-horses`,
            label: "上位馬メモ",
            note: "評価理由",
            active: activeHash === "#race-detail-agent-horses",
          },
          {
            href: `${normalizedDetailHref}#race-detail-result`,
            label: "レース結果",
            note: "確定着順",
            active: activeHash === "#race-detail-result",
          },
        ]
        : [
          {
            href: `${normalizedDetailHref}#race-detail-summary`,
            label: "レース概要",
            note: "基本情報と結果",
            active: !activeHash || activeHash === "#race-detail-summary",
          },
          {
            href: `${normalizedDetailHref}#race-detail-compare`,
            label: "定量比較",
            note: "定量モデルの本命比較",
            active: activeHash === "#race-detail-compare" || activeHash === "#race-detail-models",
          },
          {
            href: `${normalizedDetailHref}#race-detail-index`,
            label: "AI指数",
            note: "上位候補ランキング",
            active: activeHash === "#race-detail-index",
          },
        ]
      : [];

  const currentPageLabel =
    mode === "detail"
      ? detailTitle || "レース詳細"
    : mode === "history"
        ? "履歴分析"
        : mode === "reports" || mode === "reportDetail"
          ? "私の日報"
        : mode === "static"
          ? "インフォメーション"
          : "トップページ";

  const focusText =
    mode === "detail"
      ? isAgentMode
        ? "このレースの予測メモ、上位馬評価、買い目判断、結果を一つの流れで確認できます。"
        : "このレースのAI本命、AI指数、定量比較、結果を一つの流れで確認できます。"
      : mode === "history"
        ? isAgentMode
          ? "過去日付のAI予測と結果をレース単位で確認できます。"
          : "公開予測の結果と命中率を期間ごとに確認できます。"
        : mode === "reports"
          ? "保存済みの日報を一覧で確認し、対象日ごとの振り返りを読み返せます。"
        : mode === "reportDetail"
            ? "AI予測、結果、振り返りを記事形式で確認できます。"
        : mode === "static"
          ? "サイトの考え方と利用上の案内をまとめています。"
          : isAgentMode
            ? "対象日の公開レースを場別に確認し、気になるレースの予測メモへ進めます。"
            : "見どころ、深掘り分析、公開レースの順に読み進められます。";

  return (
    <aside className="public-side-nav" aria-label="サイトナビゲーション">
      <div className="public-side-nav__panel">
        <div className="public-side-nav__brand">
          <span className="public-side-nav__eyebrow">Navigation</span>
          <strong className="public-side-nav__title">いかいもAI競馬</strong>
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

        {shouldShowTargetFilter ? (
          <div className="public-side-nav__section public-side-nav__section--filter">
            <div className="public-side-nav__filter-intro">
              <span className="public-side-nav__filter-title">
                レース一覧
              </span>
              <p className="public-side-nav__filter-note">
                日付を選ぶと、その日の公開レース一覧へ移動できます。
              </p>
            </div>
            <FilterBar
              data={data}
              search={search}
              onApply={onApplyFilters}
              className="app-filter-bar--sidebar"
              context="sidebar"
            />
          </div>
        ) : null}

        {detailItems.length ? (
          <div className="public-side-nav__section">
            <span className="public-side-nav__section-label">詳細案内</span>
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

        <div className="public-side-nav__focus">
          <span className="public-side-nav__focus-label">現在表示中</span>
          <strong>{currentPageLabel}</strong>
          <p>{focusText}</p>
        </div>
      </div>
    </aside>
  );
}

