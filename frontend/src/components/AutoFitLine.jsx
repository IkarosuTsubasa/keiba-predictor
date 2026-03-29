import React, { useLayoutEffect, useRef } from "react";

export default function AutoFitLine({
  as: Tag = "span",
  className = "",
  children,
  maxFontSize = 16,
  minFontSize = 12,
}) {
  const ref = useRef(null);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    let frameId = 0;
    let observer = null;

    const fit = () => {
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
