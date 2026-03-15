import React, { useEffect, useMemo, useState } from "react";
import AppHeader from "./components/AppHeader";
import EmptyRaceState from "./components/EmptyRaceState";
import PageSectionHeader from "./components/PageSectionHeader";
import RaceGrid, { sortRacesForDisplay } from "./components/RaceGrid";
import SecondaryStatsPanel from "./components/SecondaryStatsPanel";

const APP_BASE_PATH = "/keiba";
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;

function buildQuery(search) {
  return search ? `?${search}` : "";
}

function navigateWithSearch(nextSearch) {
  const url = nextSearch ? `${APP_BASE_PATH}?${nextSearch}` : APP_BASE_PATH;
  window.history.pushState({}, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
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

    fetch(`${PUBLIC_BOARD_API_PATH}${buildQuery(search)}`, {
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

function LoadingState() {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel">
        <span className="public-screen-state__eyebrow">Loading</span>
        <h1>本日の予想一覧を読み込み中</h1>
        <p>公開中のレースと AI 予想を取得しています。</p>
      </section>
    </main>
  );
}

function ErrorState({ error, onRetry }) {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel public-screen-state__panel--error">
        <span className="public-screen-state__eyebrow">Error</span>
        <h1>公開データを表示できませんでした</h1>
        <p>{error || "時間をおいてから再読み込みしてください。"}</p>
        <button type="button" onClick={onRetry}>
          再読み込み
        </button>
      </section>
    </main>
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
  const races = useMemo(() => sortRacesForDisplay(data?.races || []), [data]);

  if (loading) {
    return <LoadingState />;
  }

  if (error || !data) {
    return <ErrorState error={error} onRetry={() => setSearch(window.location.search.replace(/^\?/, ""))} />;
  }

  return (
    <main className="racing-intel-page">
      <AppHeader data={data} search={search} onApplyFilters={navigateWithSearch} />

      <div className="racing-intel-page__shell">
        <section className="today-races-section">
          <PageSectionHeader
            kicker="Today Races"
            title="今日のAI競馬予想"
            subtitle="複数 AI の本命と買い目を一覧で確認"
            meta={[
              data?.target_date_label || "-",
              `${races.length} レース`,
              data?.generated_at_label ? `最終更新 ${data.generated_at_label}` : "",
            ]}
          />

          {data.fallback_notice ? <section className="notice-strip">{data.fallback_notice}</section> : null}

          {races.length ? <RaceGrid races={races} /> : <EmptyRaceState />}
        </section>

      </div>

      <SecondaryStatsPanel data={data} />
    </main>
  );
}
