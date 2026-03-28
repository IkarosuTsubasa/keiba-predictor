import React, { useEffect, useState } from "react";

const MOBILE_BREAKPOINT = 760;
const DESKTOP_BANNER = {
  key: "5af93c2151d30116a558aff94c470657",
  width: 728,
  height: 90,
};
const MOBILE_BANNER = {
  key: "295d20892d9f02149cc7e215520a9613",
  width: 320,
  height: 50,
};

function resolveIsMobile() {
  if (typeof window === "undefined") {
    return false;
  }
  return window.innerWidth < MOBILE_BREAKPOINT;
}

function AdsterraBannerSlot({ slotKey, width, height }) {
  useEffect(() => {
    if (typeof document === "undefined") {
      return undefined;
    }

    const target = document.getElementById(`footer-adsterra-${slotKey}`);
    if (!target) {
      return undefined;
    }

    target.innerHTML = "";

    const optionsScript = document.createElement("script");
    optionsScript.type = "text/javascript";
    optionsScript.text = `atOptions = ${JSON.stringify({
      key: slotKey,
      format: "iframe",
      height,
      width,
      params: {},
    })};`;

    const invokeScript = document.createElement("script");
    invokeScript.type = "text/javascript";
    invokeScript.async = true;
    invokeScript.src = `https://www.highperformanceformat.com/${slotKey}/invoke.js`;

    target.appendChild(optionsScript);
    target.appendChild(invokeScript);

    return () => {
      target.innerHTML = "";
    };
  }, [height, slotKey, width]);

  return (
    <div
      className="site-footer__ad-slot"
      style={{ "--ad-width": `${width}px`, "--ad-height": `${height}px` }}
    >
      <div id={`footer-adsterra-${slotKey}`} />
    </div>
  );
}

export default function FooterAdBanner() {
  const [isMobile, setIsMobile] = useState(resolveIsMobile);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const onResize = () => setIsMobile(resolveIsMobile());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const banner = isMobile ? MOBILE_BANNER : DESKTOP_BANNER;

  return (
    <div className="site-footer__ad-shell" aria-label="スポンサーリンク">
      <AdsterraBannerSlot
        slotKey={banner.key}
        width={banner.width}
        height={banner.height}
      />
    </div>
  );
}
