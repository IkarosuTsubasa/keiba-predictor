import React, { useEffect, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import BeginnerGuideSection from "./components/BeginnerGuideSection";
import DailyReportDetailPage from "./components/DailyReportDetailPage";
import DailyReportsPage from "./components/DailyReportsPage";
import FeaturedContentSection from "./components/FeaturedContentSection";
import HeroSpotlightStrip from "./components/HeroSpotlightStrip";
import HistoryPage from "./components/HistoryPage";
import HomeHeroSection from "./components/HomeHeroSection";
import MethodSummarySection from "./components/MethodSummarySection";
import PageSectionHeader from "./components/PageSectionHeader";
import PublicSideNav from "./components/PublicSideNav";
import PublicStaticPage from "./components/PublicStaticPage";
import RaceDetailPage from "./components/RaceDetailPage";
import SecondaryStatsPanel from "./components/SecondaryStatsPanel";
import SiteFooter from "./components/SiteFooter";
import TodayBoardContent from "./components/TodayBoardContent";
import { trackPageView } from "./lib/analytics";
import { buildNextPredictionSummary, buildTargetDateContext } from "./lib/homepage";
import { matchRaceIdentifier } from "./lib/publicRace";
import { HOME_PAGE_TITLE, PUBLIC_PAGE_CONTENT, SITE_NAME } from "./lib/siteCopy";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;

function buildQuery(search) {
  return search ? `?${search}` : "";
}

function extractSelectedDate(search) {
  try {
    return new URLSearchParams(String(search || "")).get("date") || "";
  } catch {
    return "";
  }
}

function navigateWithSearch(nextSearch) {
  const url = nextSearch ? `${APP_BASE_PATH}?${nextSearch}` : APP_BASE_PATH;
  window.history.pushState({}, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function extractRaceDetailId(pathname) {
  const normalized = String(pathname || "").replace(/\/+$/, "");
  const prefix = `${APP_BASE_PATH}/race/`;
  if (!normalized.startsWith(prefix)) {
    return "";
  }
  const encodedId = normalized.slice(prefix.length);
  if (!encodedId) {
    return "";
  }
  try {
    return decodeURIComponent(encodedId);
  } catch {
    return encodedId;
  }
}

function extractReportSlug(pathname) {
  const normalized = String(pathname || "").replace(/\/+$/, "");
  const prefix = `${APP_BASE_PATH}/reports/`;
  if (!normalized.startsWith(prefix)) {
    return "";
  }
  const slug = normalized.slice(prefix.length);
  if (!slug) {
    return "";
  }
  try {
    return decodeURIComponent(slug);
  } catch {
    return slug;
  }
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
      cache: "no-store",
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
          error: error?.message || "公開データの読み込みに失敗しました。",
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
    <section className="public-screen-state">
      <section className="public-screen-state__panel">
        <span className="public-screen-state__eyebrow">読み込み中</span>
        <div className="public-screen-state__loader" aria-hidden="true">
          <span className="public-screen-state__loader-ring" />
          <span className="public-screen-state__loader-core" />
          <span className="public-screen-state__loader-pulse public-screen-state__loader-pulse--one" />
          <span className="public-screen-state__loader-pulse public-screen-state__loader-pulse--two" />
        </div>
        <h1 className="public-screen-state__loading-title">公開レースを読み込んでいます</h1>
        <p>最新の公開データを取得しています。しばらくお待ちください。</p>
      </section>
    </section>
  );
}

function ErrorState({ error, onRetry }) {
  return (
    <section className="public-screen-state">
      <section className="public-screen-state__panel public-screen-state__panel--error">
        <span className="public-screen-state__eyebrow">エラー</span>
        <h1>公開情報を表示できませんでした</h1>
        <p>{error || "読み込みに失敗しました。"}</p>
        <button type="button" onClick={onRetry}>
          再読み込み
        </button>
      </section>
    </section>
  );
}

function PublicFrame({ headerProps = {}, sideNavProps = {}, children }) {
  return (
    <main className="racing-intel-page">
      <AppHeader {...headerProps} />
      <div className="racing-intel-page__shell racing-intel-page__shell--with-sidebar">
        <PublicSideNav {...sideNavProps} />
        <div className="racing-intel-page__main">{children}</div>
      </div>
      <SiteFooter />
    </main>
  );
}

function resolvePublicSideNavMode({
  isHistoryPage,
  isReportsPage,
  isReportDetail,
  isRaceDetail,
  staticPage,
}) {
  if (isHistoryPage) {
    return "history";
  }
  if (isReportsPage) {
    return "reports";
  }
  if (isReportDetail) {
    return "reportDetail";
  }
  if (isRaceDetail) {
    return "detail";
  }
  if (staticPage) {
    return "static";
  }
  return "home";
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
  const isHistoryPage = normalizedPath === `${APP_BASE_PATH}/history`;
  const isReportsPage = normalizedPath === `${APP_BASE_PATH}/reports`;
  const reportSlug = extractReportSlug(normalizedPath);
  const isReportDetail = Boolean(reportSlug);
  const raceDetailId = extractRaceDetailId(normalizedPath);
  const isRaceDetail = Boolean(raceDetailId);
  const staticPage = PUBLIC_PAGE_CONTENT[normalizedPath] || null;
  const selectedDate = extractSelectedDate(search);
  const isDateFocusedHome = normalizedPath === APP_BASE_PATH && Boolean(selectedDate);
  useEffect(() => {
    if (isAdminWorkspace) {
      document.title = `管理ワークスペース | ${SITE_NAME}`;
      return;
    }
    if (isAdminConsole) {
      document.title = `管理コンソール | ${SITE_NAME}`;
      return;
    }
    if (isRaceDetail) {
      document.title = `レース詳細 | ${SITE_NAME}`;
      return;
    }
    if (isHistoryPage) {
      document.title = `履歴分析 | ${SITE_NAME}`;
      return;
    }
    if (isReportsPage || isReportDetail) {
      document.title = `私の日報 | ${SITE_NAME}`;
      return;
    }
    if (staticPage) {
      document.title = `${staticPage.title} | ${SITE_NAME}`;
      return;
    }
    document.title = HOME_PAGE_TITLE;
  }, [isAdminConsole, isAdminWorkspace, isHistoryPage, isRaceDetail, isReportsPage, isReportDetail, staticPage]);

  useEffect(() => {
    if (isAdminConsole || isAdminWorkspace) {
      return;
    }

    const pagePath = search ? `${normalizedPath}?${search}` : normalizedPath;
    trackPageView(pagePath, document.title);
  }, [isAdminConsole, isAdminWorkspace, normalizedPath, search]);

  const shouldLoadPublicHeaderData = !isAdminConsole && !isAdminWorkspace;
  const { data: headerData } = useBoardData("", shouldLoadPublicHeaderData);
  const { loading, error, data } = useBoardData(
    search,
    !isAdminConsole && !isAdminWorkspace && !staticPage && !isReportsPage && !isReportDetail,
  );
  const targetDateContext = data ? buildTargetDateContext(data) : null;
  const nextPredictionSource = headerData || data;
  const nextPrediction = nextPredictionSource
    ? buildNextPredictionSummary(nextPredictionSource)
    : null;
  const publicHeaderProps = { showFilters: false, nextPrediction };
  const basePublicSideNavProps = {
    pathname: normalizedPath,
    mode: resolvePublicSideNavMode({
      isHistoryPage,
      isReportsPage,
      isReportDetail,
      isRaceDetail,
      staticPage,
    }),
  };
  const races = data?.races || [];
  const selectedRace = isRaceDetail
    ? races.find((race) => matchRaceIdentifier(race, raceDetailId))
    : null;

  useEffect(() => {
    const shouldHideStaticIntro =
      normalizedPath === APP_BASE_PATH && !loading && !error && Boolean(data);
    document.body.classList.toggle("home-app-ready", shouldHideStaticIntro);
    return () => document.body.classList.remove("home-app-ready");
  }, [data, error, loading, normalizedPath]);

  if (isAdminWorkspace) {
    return <AdminWorkspacePage appBasePath={APP_BASE_PATH} />;
  }

  if (isAdminConsole) {
    return <AdminJobsPage appBasePath={APP_BASE_PATH} />;
  }

  if (staticPage) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{ pathname: normalizedPath, mode: "static" }}
      >
        <div className="public-content-stack">
          <PublicStaticPage page={staticPage} />
        </div>
      </PublicFrame>
    );
  }

  if (isReportsPage) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "reports",
        }}
      >
        <div className="public-content-stack">
          <DailyReportsPage appBasePath={APP_BASE_PATH} />
        </div>
      </PublicFrame>
    );
  }

  if (isReportDetail) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "reportDetail",
        }}
      >
        <div className="public-content-stack">
          <DailyReportDetailPage slug={reportSlug} appBasePath={APP_BASE_PATH} />
        </div>
      </PublicFrame>
    );
  }

  if (loading) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          ...basePublicSideNavProps,
          detailHref: isRaceDetail ? `${normalizedPath}${buildQuery(search)}` : "",
        }}
      >
        <LoadingState />
      </PublicFrame>
    );
  }

  if (error || !data) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          ...basePublicSideNavProps,
          detailHref: isRaceDetail ? `${normalizedPath}${buildQuery(search)}` : "",
        }}
      >
        <ErrorState
          error={error}
          onRetry={() => setSearch(window.location.search.replace(/^\?/, ""))}
        />
      </PublicFrame>
    );
  }

  if (isHistoryPage) {
    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "history",
          data,
          search,
          onApplyFilters: navigateWithSearch,
          showTargetFilter: true,
        }}
      >
        <div className="public-content-stack">
          <HistoryPage data={data} />
        </div>
      </PublicFrame>
    );
  }

  if (isRaceDetail) {
    if (!selectedRace) {
      return (
        <PublicFrame
          headerProps={publicHeaderProps}
          sideNavProps={{
            pathname: normalizedPath,
            mode: "detail",
            detailHref: `${normalizedPath}${buildQuery(search)}`,
            data,
            search,
            onApplyFilters: navigateWithSearch,
            showTargetFilter: true,
          }}
        >
          <div className="public-content-stack">
            <section className="empty-race-state">
              <span className="empty-race-state__eyebrow">レース詳細</span>
              <h2>対象のレースが見つかりません</h2>
              <p>
                一覧ページに戻って、対象日や公開状況を確認してから再度お試しください。
              </p>
              <a
                className="race-detail-back-link race-detail-back-link--inline"
                href={search ? `${APP_BASE_PATH}?${search}` : APP_BASE_PATH}
              >
                一覧へ戻る
              </a>
            </section>
          </div>
        </PublicFrame>
      );
    }

    return (
      <PublicFrame
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "detail",
          detailHref: `${normalizedPath}${buildQuery(search)}`,
          detailTitle:
            selectedRace?.display_header?.detail_title ||
            selectedRace?.display_header?.title ||
            "レース詳細",
          data,
          search,
          onApplyFilters: navigateWithSearch,
          showTargetFilter: true,
        }}
      >
        <div className="public-content-stack">
          <RaceDetailPage race={selectedRace} search={search} />
        </div>
      </PublicFrame>
    );
  }

  return (
    <PublicFrame
      headerProps={publicHeaderProps}
      sideNavProps={{
        pathname: normalizedPath,
        mode: "home",
        data,
        search,
        onApplyFilters: navigateWithSearch,
        showTargetFilter: true,
      }}
    >
      <div className="public-content-stack public-content-stack--home">
        {!isDateFocusedHome ? (
          <>
            <HomeHeroSection data={data} search={search} />
            <FeaturedContentSection data={data} />
            <MethodSummarySection />
          </>
        ) : null}

        <section className="today-races-section" id="home-race-board">
          <PageSectionHeader
            kicker="公開レース"
            title={targetDateContext?.raceBoardTitle || "対象日の公開レース"}
            subtitle="比較用の導読を確認したあとに、各レースの印、買い目、結果、回収率をレース単位とモデル単位の両方から見比べられます。"
            actions={
              data?.daily_report?.public_url
                ? [
                    {
                      href: data.daily_report.public_url,
                      label: "日報を見る",
                    },
                  ]
                : []
            }
            meta={[
              {
                key: "target-date",
                label: "対象日",
                value: data?.target_date_label || "-",
              },
              {
                key: "race-count",
                label: "公開数",
                value: `${races.length}レース`,
              },
            ]}
          />

          {!isDateFocusedHome ? <HeroSpotlightStrip data={data} /> : null}
          <TodayBoardContent data={data} races={races} />
        </section>

        <SecondaryStatsPanel data={data} />
        {!isDateFocusedHome ? <BeginnerGuideSection /> : null}
      </div>
    </PublicFrame>
  );
}

