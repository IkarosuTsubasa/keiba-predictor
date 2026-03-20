import React from "react";

const FOOTER_LINKS = [
  { href: "/keiba", label: "予想ボード" },
  { href: "/keiba/about", label: "このサイトについて" },
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
          本サイトの掲載内容は情報提供を目的としたものであり、投票結果や利益を保証するものではありません。
        </p>
      </div>
    </footer>
  );
}
