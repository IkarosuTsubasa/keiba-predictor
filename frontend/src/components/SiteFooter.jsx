import React from "react";

const FOOTER_LINKS = [
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
          <strong className="site-footer__title">いかいもAI競馬</strong>
          <p className="site-footer__note">
            公開レースの予測、単場詳細、履歴分析をまとめて確認できる競馬予測サイトです。
          </p>
        </div>

        <nav className="site-footer__links" aria-label="フッターナビゲーション">
          {FOOTER_LINKS.map((item) => (
            <a key={item.href} href={item.href}>
              {item.label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
