import React, { useEffect, useState } from "react";

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
}) {
  const [activeHash, setActiveHash] = useState("");
  const normalizedPath = normalizePath(pathname);
  const normalizedDetailHref = detailHref || pathname;

  useEffect(() => {
    const syncHash = () => {
      setActiveHash(window.location.hash || "");
    };

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
      label: "履歴情報",
      note: "30日・365日・累計",
      active: normalizedPath === "/keiba/history",
    },
  ];

  const detailItems =
    mode === "detail"
      ? [
          {
            href: `${normalizedDetailHref}#race-detail-summary`,
            label: "レース概要",
            note: "開催情報と要点",
            active: !activeHash || activeHash === "#race-detail-summary",
          },
          {
            href: `${normalizedDetailHref}#race-detail-compare`,
            label: "推奨馬比較",
            note: "各 LLM の本命印",
            active: activeHash === "#race-detail-compare",
          },
          {
            href: `${normalizedDetailHref}#race-detail-models`,
            label: "購入プラン",
            note: "モデル別の買い目",
            active: activeHash === "#race-detail-models",
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
          : "公開予測ボード";

  const focusText =
    mode === "detail"
      ? "左の 3 セクションから、レース詳細をすばやく移動できます。"
      : mode === "static"
        ? "公開ページの主要導線に戻りやすい補助ナビとして配置しています。"
        : "公開ページの主要導線を左側に固定しています。";

  return (
    <aside className="public-side-nav" aria-label="公開ナビゲーション">
      <div className="public-side-nav__panel">
        <div className="public-side-nav__brand">
          <span className="public-side-nav__eyebrow">公開分析デスク</span>
          <strong className="public-side-nav__title">競馬インテリジェンス</strong>
          <p className="public-side-nav__lead">
            予測一覧と履歴分析を同じ導線で切り替えられる公開ビューです。
          </p>
        </div>

        <div className="public-side-nav__section">
          <span className="public-side-nav__section-label">主要ナビゲーション</span>
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
            <span className="public-side-nav__section-label">レース詳細</span>
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
