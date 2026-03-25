import React from "react";

const FOOTER_LINKS = [
  { href: "/keiba", label: "ホーム" },
  { href: "/keiba/history", label: "履歴分析" },
  { href: "/keiba/about", label: "このサイトについて" },
  { href: "/keiba/guide", label: "ガイド" },
  { href: "/keiba/methodology", label: "分析方針" },
  { href: "/keiba/privacy", label: "プライバシーポリシー" },
  { href: "/keiba/terms", label: "利用規約" },
  { href: "/keiba/disclaimer", label: "免責事項" },
  { href: "/keiba/contact", label: "お問い合わせ" },
];

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <div className="site-footer__brand">
          <span className="site-footer__eyebrow">公開インフォメーション</span>
          <strong className="site-footer__title">いかいもAI競馬</strong>
        </div>
        <nav className="site-footer__links" aria-label="フッターナビゲーション">
          {FOOTER_LINKS.map((item) => (
            <a key={item.href} href={item.href}>
              {item.label}
            </a>
          ))}
        </nav>
        <p className="site-footer__note">
          本サイトの公開情報は参考情報です。表示内容は随時更新されるため、最終的な判断はご自身でご確認ください。
        </p>
      </div>
    </footer>
  );
}
