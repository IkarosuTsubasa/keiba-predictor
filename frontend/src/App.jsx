import React, { useEffect, useMemo, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import EmptyRaceState from "./components/EmptyRaceState";
import PageSectionHeader from "./components/PageSectionHeader";
import RaceGrid, { sortRacesForDisplay } from "./components/RaceGrid";
import SecondaryStatsPanel from "./components/SecondaryStatsPanel";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;

function buildQuery(search) {
  return search ? `?${search}` : "";
}

function navigateWithSearch(nextSearch) {
  const url = nextSearch ? `${APP_BASE_PATH}?${nextSearch}` : APP_BASE_PATH;
  window.history.pushState({}, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function useBoardData(search, enabled = true) {
  const [state, setState] = useState({
    loading: enabled,
    error: "",
    data: null,
  });

  useEffect(() => {
    if (!enabled) {
      setState({ loading: false, error: "", data: null });
      return;
    }

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
          error: error?.message || "公開データの取得に失敗しました。",
          data: null,
        });
      });

    return () => {
      alive = false;
    };
  }, [enabled, search]);

  return state;
}

function LoadingState() {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel">
        <span className="public-screen-state__eyebrow">Loading</span>
        <h1>本日の公開レースを読み込み中</h1>
        <p>公開 board API から最新データを取得しています。</p>
      </section>
    </main>
  );
}

function ErrorState({ error, onRetry }) {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel public-screen-state__panel--error">
        <span className="public-screen-state__eyebrow">Error</span>
        <h1>公開予想を表示できません</h1>
        <p>{error || "再読み込みしてください。"}</p>
        <button type="button" onClick={onRetry}>
          再読み込み
        </button>
      </section>
    </main>
  );
}

export default function App() {
  const [pathname, setPathname] = useState(window.location.pathname);
  const [search, setSearch] = useState(window.location.search.replace(/^\?/, ""));

  useEffect(() => {
    const onPop = () => {
      setPathname(window.location.pathname);
      setSearch(window.location.search.replace(/^\?/, ""));
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const normalizedPath = String(pathname || "").replace(/\/+$/, "") || "/";
  const isAdminWorkspace = normalizedPath === ADMIN_WORKSPACE_PATH;
  const isAdminConsole = normalizedPath === ADMIN_CONSOLE_PATH;

  useEffect(() => {
    if (isAdminWorkspace) {
      document.title = "Workspace | いかいもAI競馬";
    } else if (isAdminConsole) {
      document.title = "管理コンソール | いかいもAI競馬";
    } else {
      document.title = "いかいもAI競馬";
    }
  }, [isAdminConsole, isAdminWorkspace]);

  const { loading, error, data } = useBoardData(search, !isAdminConsole && !isAdminWorkspace);
  const races = useMemo(() => sortRacesForDisplay(data?.races || []), [data]);

  if (isAdminWorkspace) {
    return <AdminWorkspacePage appBasePath={APP_BASE_PATH} />;
  }

  if (isAdminConsole) {
    return <AdminJobsPage appBasePath={APP_BASE_PATH} />;
  }

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
            kicker="TODAY'S RACES"
            title="本日のAI予想"
            subtitle="公開ページでは各レースの印と買い目を一覧で確認できます。"
            meta={[
              data?.target_date_label || "-",
              `${races.length}レース`,
              data?.generated_at_label ? `更新 ${data.generated_at_label}` : "",
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
