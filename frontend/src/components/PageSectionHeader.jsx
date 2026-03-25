import React, { useLayoutEffect, useRef } from "react";

function normalizeMetaItem(item) {
  if (item && typeof item === "object") {
    return {
      key: item.key || `${item.label || ""}-${item.value || ""}`,
      label: item.label || "",
      value: item.value || "",
    };
  }
  return {
    key: String(item),
    label: "",
    value: String(item || ""),
  };
}

function AutoFitLine({
  as: Tag = "span",
  className = "",
  children,
  maxFontSize,
  minFontSize,
}) {
  const ref = useRef(null);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    let frameId = 0;
    let observer = null;

    const fit = () => {
      if (!node) return;

      const max = Number(maxFontSize) || 16;
      const min = Number(minFontSize) || 12;
      node.style.fontSize = `${max}px`;

      if (node.scrollWidth <= node.clientWidth + 1) {
        return;
      }

      let low = min;
      let high = max;
      while (high - low > 0.25) {
        const mid = (low + high) / 2;
        node.style.fontSize = `${mid}px`;
        if (node.scrollWidth <= node.clientWidth + 1) {
          low = mid;
        } else {
          high = mid;
        }
      }

      node.style.fontSize = `${Math.max(min, Math.floor(low * 10) / 10)}px`;
    };

    const queueFit = () => {
      window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(fit);
    };

    queueFit();

    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(queueFit);
      observer.observe(node);
      if (node.parentElement) {
        observer.observe(node.parentElement);
      }
    }

    window.addEventListener("resize", queueFit);
    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener("resize", queueFit);
      observer?.disconnect();
    };
  }, [children, maxFontSize, minFontSize]);

  return (
    <Tag ref={ref} className={className}>
      {children}
    </Tag>
  );
}

export default function PageSectionHeader({
  kicker,
  title,
  subtitle = "",
  meta = [],
}) {
  const visibleMeta = (meta || [])
    .filter(Boolean)
    .map(normalizeMetaItem)
    .filter((item) => item.value && item.key !== "generated-at");

  return (
    <div className="page-section-header">
      <div className="page-section-header__copy">
        <span className="page-section-header__kicker">{kicker}</span>
        <h1>{title}</h1>
        {subtitle ? (
          <AutoFitLine
            as="p"
            className="page-section-header__subtitle"
            maxFontSize={14}
            minFontSize={11}
          >
            {subtitle}
          </AutoFitLine>
        ) : null}
      </div>
      {visibleMeta.length ? (
        <div className="page-section-header__meta" aria-label={kicker || title}>
          {visibleMeta.map((item) => (
            <article key={item.key} className="page-section-header__meta-card">
              {item.label ? (
                <small className="page-section-header__meta-label">{item.label}</small>
              ) : null}
              <AutoFitLine
                as="strong"
                className="page-section-header__meta-value"
                maxFontSize={22}
                minFontSize={13}
              >
                {item.value}
              </AutoFitLine>
            </article>
          ))}
        </div>
      ) : null}
    </div>
  );
}
