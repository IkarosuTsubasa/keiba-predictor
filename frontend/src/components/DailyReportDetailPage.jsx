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

export default function DailyReportDetailPage({ slug = "", appBasePath = "/keiba" }) {
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

  return (
    <section className="daily-report-detail-page">
      <div className="daily-report-detail-hero">
        <div className="daily-report-detail-hero__copy">
          <a className="race-detail-back-link" href={`${appBasePath}/reports`}>
            日報一覧へ戻る
          </a>
          <span className="daily-report-detail-hero__eyebrow">私の日報</span>
          <h1>{item?.title || "-"}</h1>
          <p>{item?.lead || item?.summary || "日報の本文はまだありません。"}</p>
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
