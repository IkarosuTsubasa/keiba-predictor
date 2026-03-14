import React, { useEffect, useMemo, useState } from "react";

function buildQuery(search) {
  return search ? `?${search}` : "";
}

function useBoardData(search) {
  const [state, setState] = useState({
    loading: true,
    error: "",
    data: null,
  });

  useEffect(() => {
    let alive = true;
    setState({ loading: true, error: "", data: null });

    fetch(`/api/public/board${buildQuery(search)}`, {
      headers: { Accept: "application/json" },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) return;
        setState({ loading: false, error: "", data });
      })
      .catch((error) => {
        if (!alive) return;
        setState({
          loading: false,
          error: error?.message || "データの読み込みに失敗しました。",
          data: null,
        });
      });

    return () => {
      alive = false;
    };
  }, [search]);

  return state;
}

function sortRaceCards(cards) {
  return [...(cards || [])].sort((a, b) => Number(b.profit_yen || 0) - Number(a.profit_yen || 0));
}

function Filters({ data, search, compact = false }) {
  const params = new URLSearchParams(search);

  return (
    <form
      className={`toolbar${compact ? " toolbar--inline" : ""}`}
      onSubmit={(event) => {
        event.preventDefault();
        const formData = new FormData(event.currentTarget);
        const next = new URLSearchParams();
        const date = String(formData.get("date") || "").trim();
        const scopeKey = String(formData.get("scope_key") || "").trim();

        if (date) next.set("date", date);
        if (scopeKey) next.set("scope_key", scopeKey);

        const url = next.toString() ? `/llm_today?${next.toString()}` : "/llm_today";
        window.history.pushState({}, "", url);
        window.dispatchEvent(new PopStateEvent("popstate"));
      }}
    >
      <label className="field">
        <span>日付</span>
        <input type="date" name="date" defaultValue={params.get("date") || data.target_date || ""} />
      </label>
      <label className="field">
        <span>範囲</span>
        <select name="scope_key" defaultValue={params.get("scope_key") || data.scope_key || ""}>
          {(data.scope_options || []).map((item) => (
            <option key={item.value || "all"} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
      <button type="submit">更新</button>
    </form>
  );
}

function Topbar({ data, search }) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <span className="topbar__eyebrow">PUBLIC BOARD</span>
        <strong>いかいも競馬AI</strong>
      </div>
      <div className="topbar__controls">
        <Filters data={data} search={search} compact />
      </div>
    </header>
  );
}

function ModelRow({ card }) {
  return (
    <article className="model-row">
      <div className="model-row__engine">
        <strong>{card.label}</strong>
      </div>
      <section className="model-row__marks">
        <p>{card.marks_text || "-"}</p>
      </section>
      <section className="model-row__tickets">
        <p>{card.ticket_plan_text || "-"}</p>
      </section>
      <div className="model-row__result">
        <span>{card.result_triplet_text || "-"}</span>
      </div>
    </article>
  );
}

function RaceBoards({ races }) {
  if (!races.length) {
    return <div className="empty-panel">この日の公開データはまだありません。</div>;
  }

  return (
    <section className="race-list">
      {races.map((race, index) => {
        const sortedCards = sortRaceCards(race.cards || []);

        return (
          <section key={`${race.run_id}-${index}`} className="race-card">
            <div className="race-card__summary">
              <div className="race-card__title">
                <strong>{race.race_title}</strong>
                <div className="race-card__meta">
                  <span>{race.actual_text || "結果未登録"}</span>
                </div>
              </div>
            </div>

            <div className="race-card__body">
              <div className="model-table">
                <div className="model-table__head">
                  <span>モデル</span>
                  <span>印</span>
                  <span>買い目</span>
                  <span>結果</span>
                </div>
                {sortedCards.map((card) => (
                  <ModelRow key={`${race.run_id}-${card.engine}`} card={card} />
                ))}
              </div>
            </div>
          </section>
        );
      })}
    </section>
  );
}

export default function App() {
  const [search, setSearch] = useState(window.location.search.replace(/^\?/, ""));

  useEffect(() => {
    document.title = "いかいも競馬AI";
  }, []);

  useEffect(() => {
    const onPop = () => setSearch(window.location.search.replace(/^\?/, ""));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const { loading, error, data } = useBoardData(search);
  const races = useMemo(() => data?.races || [], [data]);

  if (loading) {
    return (
      <main className="screen-state">
        <section className="screen-state__card">
          <div className="screen-state__badge">Loading</div>
          <h1>公開ボードを読み込み中です</h1>
          <p>印と買い目を整理しています。</p>
        </section>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="screen-state">
        <section className="screen-state__card screen-state__card--error">
          <div className="screen-state__badge">Error</div>
          <h1>公開ページを読み込めませんでした</h1>
          <p>{error || "想定外のエラーが発生しました。"}</p>
          <button type="button" onClick={() => setSearch(window.location.search.replace(/^\?/, ""))}>
            再読み込み
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="page">
      <Topbar data={data} search={search} />
      {data.fallback_notice ? <section className="notice-banner">{data.fallback_notice}</section> : null}
      <RaceBoards races={races} />
    </main>
  );
}
