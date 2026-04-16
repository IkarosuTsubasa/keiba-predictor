import React, { Fragment, useEffect, useMemo, useState } from "react";
function EmptyState({ children }) {
  return <p className="daily-report-empty-note">{children}</p>;
}

function ErrorState({ title = "", details = [] }) {
  const lines = (details || []).filter(Boolean);
  return (
    <div className="daily-report-error">
      <p className="daily-report-error__title">{title}</p>
      {lines.length ? (
        <pre className="daily-report-error__details">{lines.join("\n")}</pre>
      ) : null}
    </div>
  );
}

function buildReportShareText({ title = "", url = "" }) {
  return [
    "いかいもAI競馬",
    String(title || "").trim() || "私の日報",
    String(url || "").trim(),
    "#競馬 #AI競馬",
  ]
    .filter(Boolean)
    .join("\n");
}

function renderInline(text, keyPrefix = "inline") {
  const source = String(text || "");
  if (!source) return "";
  const parts = source.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (/^`[^`]+`$/.test(part)) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

function parseMarkdown(markdown) {
  const text = String(markdown || "").replace(/\r\n/g, "\n").trim();
  if (!text) return [];
  const lines = text.split("\n");
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const current = lines[index].trimEnd();
    if (!current.trim()) {
      index += 1;
      continue;
    }
    if (current.startsWith("```")) {
      const language = current.slice(3).trim();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trimStart().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({ type: "code", language, text: codeLines.join("\n") });
      continue;
    }
    if (/^###\s+/.test(current)) {
      blocks.push({ type: "h3", text: current.replace(/^###\s+/, "").trim() });
      index += 1;
      continue;
    }
    if (/^##\s+/.test(current) || /^#\s+/.test(current)) {
      blocks.push({ type: "h2", text: current.replace(/^#{1,2}\s+/, "").trim() });
      index += 1;
      continue;
    }
    if (/^>\s+/.test(current)) {
      const items = [];
      while (index < lines.length && /^>\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^>\s+/, ""));
        index += 1;
      }
      blocks.push({ type: "quote", text: items.join(" ") });
      continue;
    }
    if (/^[-*]\s+/.test(current)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }
    if (/^\d+\.\s+/.test(current)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push({ type: "ol", items });
      continue;
    }
    const paragraphs = [];
    while (index < lines.length) {
      const line = lines[index].trim();
      if (!line) {
        break;
      }
      if (
        line.startsWith("```") ||
        /^#{1,3}\s+/.test(line) ||
        /^>\s+/.test(line) ||
        /^[-*]\s+/.test(line) ||
        /^\d+\.\s+/.test(line)
      ) {
        break;
      }
      paragraphs.push(line);
      index += 1;
    }
    blocks.push({ type: "p", text: paragraphs.join(" ") });
  }

  return blocks;
}

function MarkdownArticle({ markdown = "" }) {
  const blocks = useMemo(() => parseMarkdown(markdown), [markdown]);
  if (!blocks.length) {
    return <EmptyState>本文はまだありません。</EmptyState>;
  }
  return (
    <div className="daily-report-markdown">
      {blocks.map((block, index) => {
        if (block.type === "h2") {
          return <h2 key={`md-${index}`}>{renderInline(block.text, `h2-${index}`)}</h2>;
        }
        if (block.type === "h3") {
          return <h3 key={`md-${index}`}>{renderInline(block.text, `h3-${index}`)}</h3>;
        }
        if (block.type === "ul") {
          return (
            <ul key={`md-${index}`} className="daily-report-article__bullets">
              {(block.items || []).map((item, itemIndex) => (
                <li key={`md-${index}-${itemIndex}`}>{renderInline(item, `ul-${index}-${itemIndex}`)}</li>
              ))}
            </ul>
          );
        }
        if (block.type === "ol") {
          return (
            <ol key={`md-${index}`} className="daily-report-article__bullets">
              {(block.items || []).map((item, itemIndex) => (
                <li key={`md-${index}-${itemIndex}`}>{renderInline(item, `ol-${index}-${itemIndex}`)}</li>
              ))}
            </ol>
          );
        }
        if (block.type === "quote") {
          return <blockquote key={`md-${index}`}>{renderInline(block.text, `quote-${index}`)}</blockquote>;
        }
        if (block.type === "code") {
          return (
            <pre key={`md-${index}`} className="daily-report-markdown__code">
              <code>{block.text}</code>
            </pre>
          );
        }
        return <p key={`md-${index}`}>{renderInline(block.text, `p-${index}`)}</p>;
      })}
    </div>
  );
}

async function parseJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function withAppShellFlag(href, enabled = false) {
  const value = String(href || "").trim();
  if (!enabled || !value) return value;
  try {
    const base = typeof window !== "undefined" ? window.location.origin : "https://www.ikaimo-ai.com";
    const url = new URL(value, base);
    url.searchParams.set("app", "1");
    return `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return value.includes("?") ? `${value}&app=1` : `${value}?app=1`;
  }
}

export default function DailyReportDetailPage({ slug = "", appBasePath = "/keiba", appShell = false }) {
  const [state, setState] = useState({
    loading: true,
    errorTitle: "",
    errorDetails: [],
    item: null,
  });

  useEffect(() => {
    let alive = true;
    const url = `${appBasePath}/api/public/reports/${encodeURIComponent(slug)}`;
    setState({ loading: true, errorTitle: "", errorDetails: [], item: null });
    fetch(url, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then(async (response) => {
        const payload = await parseJsonSafely(response);
        if (!response.ok) {
          const backendError = String(payload?.error || "").trim();
          const title = response.status === 404 ? "日報が見つかりません。" : "日報の読み込みに失敗しました。";
          throw {
            title,
            details: [
              `HTTP ${response.status}`,
              backendError ? `error: ${backendError}` : "",
              `url: ${url}`,
            ].filter(Boolean),
          };
        }
        return payload;
      })
      .then((payload) => {
        if (!alive) return;
        setState({
          loading: false,
          errorTitle: "",
          errorDetails: [],
          item: payload?.item || null,
        });
      })
      .catch((error) => {
        if (!alive) return;
        setState({
          loading: false,
          errorTitle: String(error?.title || error?.message || "日報の読み込みに失敗しました。"),
          errorDetails: Array.isArray(error?.details) ? error.details : [`url: ${url}`],
          item: null,
        });
      });
    return () => {
      alive = false;
    };
  }, [appBasePath, slug]);

  useEffect(() => {
    const title = String(state.item?.title || "").trim();
    if (title) {
      document.title = `${title} | いかいもAI競馬`;
    }
  }, [state.item]);

  if (state.loading) {
    return (
      <section className="daily-report-detail-page">
        <EmptyState>日報を読み込んでいます。</EmptyState>
      </section>
    );
  }

  if (state.errorTitle || !state.item) {
    return (
      <section className="daily-report-detail-page">
        <ErrorState title={state.errorTitle || "日報が見つかりません。"} details={state.errorDetails} />
      </section>
    );
  }

  const item = state.item || {};
  const tags = Array.isArray(item?.tags) ? item.tags : [];
  const markdown = String(item?.markdown || "").trim();
  const shareUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}${appBasePath}/reports/${encodeURIComponent(slug)}`
      : `${appBasePath}/reports/${encodeURIComponent(slug)}`;
  const shareText = buildReportShareText({
    title: item?.title || "私の日報",
    url: shareUrl,
  });

  const handleShare = async () => {
    const text = String(shareText || "").trim();
    if (!text) return;
    const isMobileShare =
      (typeof navigator !== "undefined" &&
        /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "")) ||
      (typeof window !== "undefined" &&
        window.matchMedia &&
        window.matchMedia("(max-width: 760px)").matches) ||
      (typeof window !== "undefined" && "ontouchstart" in window);

    if (isMobileShare && typeof navigator !== "undefined" && typeof navigator.share === "function") {
      try {
        await navigator.share({ text });
        return;
      } catch {
        // Fallback to X intent below.
      }
    }
    const intentUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
    if (typeof window === "undefined") {
      return;
    }
    if (isMobileShare) {
      window.location.href = intentUrl;
      return;
    }
    const width = 720;
    const height = 640;
    const left = Math.max(0, Math.round((window.screen.width - width) / 2));
    const top = Math.max(0, Math.round((window.screen.height - height) / 2));
    const popup = window.open(
      intentUrl,
      "ikaimo-share",
      `popup=yes,width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`,
    );
    if (popup && !popup.closed) {
      try {
        popup.focus();
      } catch {
        // Ignore popup focus errors.
      }
      return;
    }
    window.location.href = intentUrl;
  };

  return (
    <section className="daily-report-detail-page">
      <div className="daily-report-detail-hero">
        <div className="daily-report-detail-hero__copy">
          <div className="daily-report-detail-hero__actions">
            <a className="race-detail-back-link" href={withAppShellFlag(`${appBasePath}/reports`, appShell)}>
              日報一覧へ戻る
            </a>
            <button
              type="button"
              className="daily-report-share-button"
              onClick={handleShare}
              aria-label="Xでシェア"
              title="Xでシェア"
            >
              <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" aria-hidden="true" focusable="false">
                <path d="M18.901 1.153h3.68l-8.04 9.19L24 22.847h-7.406l-5.8-7.584-6.636 7.584H.478l8.6-9.83L0 1.153h7.594l5.243 6.932 6.064-6.932Zm-1.29 19.494h2.04L6.486 3.24H4.298l13.313 17.407Z" />
              </svg>
            </button>
          </div>
          <span className="daily-report-detail-hero__eyebrow">私の日報</span>
          <h1>{renderInline(item?.title || "-", "hero-title")}</h1>
          <p>{renderInline(item?.lead || item?.summary || "日報の本文はまだありません。", "hero-lead")}</p>
          {tags.length ? (
            <div className="daily-report-detail-hero__tags">
              {tags.map((tag) => (
                <span key={`${item.slug}-${tag}`}>{tag}</span>
              ))}
            </div>
          ) : null}
        </div>

        <div className="daily-report-detail-hero__meta">
          <article className="race-detail-summary-card race-detail-summary-card--accent">
            <span>対象日</span>
            <strong>{item?.target_date_label || item?.target_date || "-"}</strong>
          </article>
        </div>
      </div>

      <article className="daily-report-article">
        {item?.summary ? (
          <section className="daily-report-article__intro">
            <p>{renderInline(item.summary, "summary")}</p>
          </section>
        ) : null}

        <MarkdownArticle markdown={markdown} />

        {item?.mode === "fallback" ? (
          <section className="daily-report-article__note">
            <span>生成メモ</span>
            <p>
              この日報はローカル要約で保存されています。
              {item?.fallback_reason ? ` 理由: ${item.fallback_reason}` : ""}
            </p>
          </section>
        ) : null}
      </article>
    </section>
  );
}
