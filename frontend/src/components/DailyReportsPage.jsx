import React, { useEffect, useState } from "react";

function formatDateTime(value) {
  const text = String(value || "").trim();
  if (!text) return "-";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function EmptyState({ children }) {
  return <p className="daily-report-empty-note">{children}</p>;
}

function ReportCard({ item, featured = false }) {
  const href = String(item?.public_url || "").trim() || "/keiba/reports";
  const className = `daily-report-card${featured ? " daily-report-card--featured" : ""}`;
  return (
    <article className={className}>
      <div className="daily-report-card__top">
        <span>{item?.target_date_label || item?.target_date || "-"}</span>
        <em>{item?.engine_label || "-"}</em>
      </div>
      <h2>{item?.title || "-"}</h2>
      <p>{item?.lead || item?.summary || "日報の要約はまだありません。"}</p>
      <div className="daily-report-card__meta">
        <span>{formatDateTime(item?.created_at)}</span>
        <span>{item?.mode === "fallback" ? "ローカル要約" : "LLM生成"}</span>
      </div>
      <a href={href}>続きを読む</a>
    </article>
  );
}

export default function DailyReportsPage({ appBasePath = "/keiba" }) {
  const [state, setState] = useState({ loading: true, error: "", items: [] });

  useEffect(() => {
    let alive = true;
    setState({ loading: true, error: "", items: [] });
    fetch(`${appBasePath}/api/public/reports`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        if (!alive) return;
        setState({
          loading: false,
          error: "",
          items: Array.isArray(payload?.items) ? payload.items : [],
        });
      })
      .catch((error) => {
        if (!alive) return;
        setState({
          loading: false,
          error: error?.message || "日報一覧の読み込みに失敗しました。",
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
          <h1>日次アーカイブ</h1>
          <p>
            対象日の AI モデル結果、定量モデルの命中傾向、振り返りコメントを
            記事形式で蓄積しています。
          </p>
        </div>
        <div className="daily-reports-hero__meta">
          <span>保存件数</span>
          <strong>{items.length}件</strong>
        </div>
      </div>

      {state.loading ? <EmptyState>日報一覧を読み込んでいます。</EmptyState> : null}
      {state.error ? <EmptyState>{state.error}</EmptyState> : null}

      {!state.loading && !state.error && latest ? (
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

      {!state.loading && !state.error ? (
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
