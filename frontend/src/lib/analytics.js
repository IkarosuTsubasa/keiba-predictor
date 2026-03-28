const GA_MEASUREMENT_ID = "G-57WH8H1359";

function isAnalyticsEnabled() {
  return typeof window !== "undefined" && Boolean(GA_MEASUREMENT_ID);
}

export function initAnalytics() {
  if (!isAnalyticsEnabled() || typeof document === "undefined" || window.gtag) {
    return;
  }

  window.dataLayer = window.dataLayer || [];
  window.gtag = function gtag() {
    window.dataLayer.push(arguments);
  };

  window.gtag("js", new Date());
  window.gtag("config", GA_MEASUREMENT_ID, {
    send_page_view: false,
  });

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);
}

export function trackPageView(path, title) {
  if (!isAnalyticsEnabled() || typeof window.gtag !== "function") {
    return;
  }

  window.gtag("event", "page_view", {
    page_title: title,
    page_path: path,
    page_location: `${window.location.origin}${path}`,
  });
}

export function trackEvent(eventName, params = {}) {
  if (!isAnalyticsEnabled() || typeof window.gtag !== "function") {
    return;
  }

  window.gtag("event", eventName, params);
}
