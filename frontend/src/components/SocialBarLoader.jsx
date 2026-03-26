import React, { useEffect } from "react";

const SOCIAL_BAR_SRC =
  "https://pl28986795.profitablecpmratenetwork.com/5d/c9/21/5dc9219459246ebdaf545057c2ec5c92.js";
const SOCIAL_BAR_SCRIPT_ID = "adsterra-social-bar-script";
const SOCIAL_BAR_STATUS_FLAG = "__adsterraSocialBarStatus";

export default function SocialBarLoader({ enabled = false }) {
  useEffect(() => {
    if (!enabled || typeof document === "undefined" || typeof window === "undefined") {
      return;
    }

    if (window[SOCIAL_BAR_STATUS_FLAG] === "loaded" || window[SOCIAL_BAR_STATUS_FLAG] === "loading") {
      return;
    }

    const existing =
      document.getElementById(SOCIAL_BAR_SCRIPT_ID) ||
      document.querySelector(`script[src="${SOCIAL_BAR_SRC}"]`);
    if (existing) {
      window[SOCIAL_BAR_STATUS_FLAG] = "loading";
      return;
    }

    const script = document.createElement("script");
    script.id = SOCIAL_BAR_SCRIPT_ID;
    script.src = SOCIAL_BAR_SRC;
    script.async = true;
    script.dataset.adNetwork = "adsterra-social-bar";
    window[SOCIAL_BAR_STATUS_FLAG] = "loading";
    script.onload = () => {
      window[SOCIAL_BAR_STATUS_FLAG] = "loaded";
    };
    script.onerror = () => {
      window[SOCIAL_BAR_STATUS_FLAG] = "";
      script.remove();
    };
    document.body.appendChild(script);
  }, [enabled]);

  return null;
}
