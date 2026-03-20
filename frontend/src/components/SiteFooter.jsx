import React from "react";

const FOOTER_LINKS = [
  { href: "/keiba", label: "公開ボード" },
  { href: "/keiba/about", label: "このサイトについて" },
  { href: "/keiba/guide", label: "ガイド" },
  { href: "/keiba/privacy", label: "プライバシーポリシー" },
  { href: "/keiba/terms", label: "利用規約" },
  { href: "/keiba/disclaimer", label: "免責事項" },
  { href: "/keiba/contact", label: "お問い合わせ" },
];

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <nav className="site-footer__links" aria-label="Site footer">
          {FOOTER_LINKS.map((item) => (
            <a key={item.href} href={item.href}>
              {item.label}
            </a>
          ))}
        </nav>
        <p className="site-footer__note">
          当サイトの競馬分析は情報提供を目的とした参考コンテンツです。最終的な判断と投票行動は、利用者ご自身の責任でお願いします。
        </p>
      </div>
    </footer>
  );
}
