import React, { useEffect } from "react";

const ADSTERRA_SCRIPT_SRC =
  "https://pl28986796.profitablecpmratenetwork.com/952706d1670cd6380a1c19aecdd5851b/invoke.js";
const ADSTERRA_CONTAINER_ID = "container-952706d1670cd6380a1c19aecdd5851b";

export default function RaceDetailNativeBanner() {
  useEffect(() => {
    const container = document.getElementById(ADSTERRA_CONTAINER_ID);
    if (!container) {
      return undefined;
    }

    container.innerHTML = "";

    const existingScript = document.querySelector(
      `script[data-adsterra-slot="${ADSTERRA_CONTAINER_ID}"]`,
    );
    if (existingScript) {
      existingScript.remove();
    }

    const script = document.createElement("script");
    script.async = true;
    script.setAttribute("data-cfasync", "false");
    script.setAttribute("data-adsterra-slot", ADSTERRA_CONTAINER_ID);
    script.src = ADSTERRA_SCRIPT_SRC;

    container.parentNode?.insertBefore(script, container);

    return () => {
      script.remove();
      container.innerHTML = "";
    };
  }, []);

  return (
    <div className="race-detail-native-banner" data-ad-slot="adsterra-native">
      <div id={ADSTERRA_CONTAINER_ID} />
    </div>
  );
}
