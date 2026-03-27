import React, { useEffect, useState } from "react";

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

function ReportCard({ item, featured = false }) {
  const href = String(item?.public_url || "").trim() || "/keiba/reports";
  const className = `daily-report-card${featured ? " daily-report-card--featured" : ""}`;
  return (
    <article className={className}>
      <div className="daily-report-card__top">
        <span>{item?.target_date_label || item?.target_date || "-"}</span>
      </div>
      <h2>{item?.title || "-"}</h2>
      <p>{item?.lead || item?.summary || "日報の要約はまだありません。"}</p>
      <div className="daily-report-card__meta">
        <span>{item?.mode === "fallback" ? "ローカル要約" : "LLM生成"}</span>
      </div>
      <a href={href}>続きを読む</a>
    </article>
  );
}

async function parseJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export default function DailyReportsPage({ appBasePath = "/keiba" }) {
  const [state, setState] = useState({
    loading: true,
    errorTitle: "",
    errorDetails: [],
    items: [],
  });

  useEffect(() => {
    let alive = true;
    const url = `${appBasePath}/api/public/reports`;
    setState({ loading: true, errorTitle: "", errorDetails: [], items: [] });
    fetch(url, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then(async (response) => {
        const payload = await parseJsonSafely(response);
        if (!response.ok) {
          const backendError = String(payload?.error || "").trim();
          throw {
            title: "日報一覧の読み込みに失敗しました。",
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
          items: Array.isArray(payload?.items) ? payload.items : [],
        });
      })
      .catch((error) => {
        if (!alive) return;
        setState({
          loading: false,
          errorTitle: String(error?.title || error?.message || "日報一覧の読み込みに失敗しました。"),
          errorDetails: Array.isArray(error?.details)
            ? error.details
            : [`url: ${url}`],
          items: [],
        });
      });
    return () => {
      alive = false;
    };
  }, [appBasePath]);

  const items = state.items || [];
  const latest = items[0] || null;

  return (
    <section className="daily-reports-page">
      <div className="daily-reports-hero">
        <div className="daily-reports-hero__copy">
          <span className="daily-reports-hero__eyebrow">私の日報</span>
          <h1>日報アーカイブ</h1>
          <p>
            対象日の AI モデル成績、予測モデルの命中傾向、回収率の振り返りコメントを
            記事形式で確認できます。
          </p>
        </div>
        <div className="daily-reports-hero__meta">
          <span>保存件数</span>
          <strong>{items.length}件</strong>
        </div>
      </div>

      {state.loading ? <EmptyState>日報一覧を読み込んでいます。</EmptyState> : null}
      {state.errorTitle ? (
        <ErrorState title={state.errorTitle} details={state.errorDetails} />
      ) : null}

      {!state.loading && !state.errorTitle && latest ? (
        <section className="daily-report-panel">
          <div className="daily-report-panel__head">
            <div>
              <span className="daily-report-panel__eyebrow">最新日報</span>
              <h2>Latest Entry</h2>
            </div>
          </div>
          <ReportCard item={latest} featured />
        </section>
      ) : null}

      {!state.loading && !state.errorTitle ? (
        <section className="daily-report-panel">
          <div className="daily-report-panel__head">
            <div>
              <span className="daily-report-panel__eyebrow">アーカイブ</span>
              <h2>Saved Reports</h2>
            </div>
          </div>
          {items.length ? (
            <div className="daily-report-grid">
              {items.map((item) => (
                <ReportCard key={item.slug || item.created_at} item={item} />
              ))}
            </div>
          ) : (
            <EmptyState>保存済みの日報はまだありません。</EmptyState>
          )}
        </section>
      ) : null}
    </section>
  );
}
