import React, { useEffect, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import BeginnerGuideSection from "./components/BeginnerGuideSection";
import DailyReportDetailPage from "./components/DailyReportDetailPage";
import DailyReportsPage from "./components/DailyReportsPage";
import FeaturedContentSection from "./components/FeaturedContentSection";
import FilterBar from "./components/FilterBar";
import HistoryPage from "./components/HistoryPage";
import HomeHeroSection from "./components/HomeHeroSection";
import MethodSummarySection from "./components/MethodSummarySection";
import MorningPreviewSection from "./components/MorningPreviewSection";
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
const { useMemo } = React;

function buildQuery(search) {
  return search ? `?${search}` : "";
}

function buildMorningBadges(item, baseRace) {
  const baseBadges = Array.isArray(baseRace?.display_header?.badges)
    ? baseRace.display_header.badges.filter(Boolean)
    : [];
  if (baseBadges.length) {
    return baseBadges;
  }
  return [
    String(item?.scheduled_off_time || "").match(/(\d{2}:\d{2})/)?.[1] || "",
    item?.distance_label || "",
    item?.track_condition || "",
  ].filter(Boolean);
}

function buildMorningPreviewRace(item, baseRace = null) {
  const title =
    String(baseRace?.display_header?.title || "").trim() ||
    String(item?.race_title || "").trim() ||
    "-";
  const subtitle =
    String(baseRace?.display_header?.subtitle || "").trim() ||
    String(item?.race_name || "").trim();
  return {
    ...(baseRace || {}),
    ...item,
    run_id: item?.run_id || baseRace?.run_id || "",
    race_id: item?.race_id || baseRace?.race_id || "",
    location: item?.location || baseRace?.location || "",
    scheduled_off_time: item?.scheduled_off_time || baseRace?.scheduled_off_time || "",
    display_order: Number.isFinite(Number(baseRace?.display_order))
      ? Number(baseRace.display_order)
      : Number.MAX_SAFE_INTEGER,
    display_variant: "morning_preview",
    display_status: {
      label: "速報",
      tone: "open",
    },
    display_header: {
      title,
      subtitle,
      detail_title: subtitle ? `${title} ${subtitle}`.trim() : title,
      badges: buildMorningBadges(item, baseRace),
    },
    display_body: {
      kind: "morning_preview",
      result_text: item?.summary_text || "速報を表示中",
    },
    cards: [],
  };
}

function mergeMorningPreviewRaces(races, preview) {
  const baseRaces = Array.isArray(races) ? races : [];
  const morningRaces = Array.isArray(preview?.races) ? preview.races : [];
  if (!preview?.available || !morningRaces.length) {
    return baseRaces;
  }

  const previewMatchKey = (race) => {
    const raceTitle = String(race?.race_title || "").trim();
    if (raceTitle) {
      return `title:${raceTitle}`;
    }
    const location = String(race?.location || "").trim();
    const raceId = String(race?.race_id || "").trim();
    if (location && raceId) {
      return `loc:${location}:${raceId}`;
    }
    if (raceId) {
      return `id:${raceId}`;
    }
    return "";
  };

  const morningByRaceId = new Map();
  for (const item of morningRaces) {
    const key = previewMatchKey(item);
    if (key) {
      morningByRaceId.set(key, item);
    }
  }

  const seen = new Set();
  const merged = baseRaces.map((race) => {
    const raceKey = previewMatchKey(race);
    const morning = morningByRaceId.get(raceKey);
    if (!morning) {
      return race;
    }
    seen.add(raceKey);
    if (String(race?.display_variant || "").trim() === "placeholder") {
      return buildMorningPreviewRace(morning, race);
    }
    return race;
  });

  for (const item of morningRaces) {
    const raceKey = previewMatchKey(item);
    if (raceKey && seen.has(raceKey)) {
      continue;
    }
    merged.push(buildMorningPreviewRace(item));
  }

  return merged;
}

function raceDisplayMatchKey(race) {
  const raceTitle = String(race?.race_title || "").trim();
  if (raceTitle) {
    return `title:${raceTitle}`;
  }
  const location = String(race?.location || "").trim();
  const raceId = String(race?.race_id || "").trim();
  if (location && raceId) {
    return `loc:${location}:${raceId}`;
  }
  if (raceId) {
    return `id:${raceId}`;
  }
  return "";
}

function resolveSelectedRace(boardRaces, raceDetailId) {
  const matchedRace = (Array.isArray(boardRaces) ? boardRaces : []).find((race) =>
    matchRaceIdentifier(race, raceDetailId),
  );
  if (!matchedRace) {
    return null;
  }
  if (String(matchedRace?.display_variant || "").trim() !== "morning_preview") {
    return matchedRace;
  }
  const matchKey = raceDisplayMatchKey(matchedRace);
  if (!matchKey) {
    return matchedRace;
  }
  const preferredRace = (Array.isArray(boardRaces) ? boardRaces : []).find((race) => {
    if (race === matchedRace) return false;
    if (raceDisplayMatchKey(race) !== matchKey) return false;
    return String(race?.display_variant || "").trim() !== "morning_preview";
  });
  return preferredRace || matchedRace;
}

function extractSelectedDate(search) {
  try {
    return new URLSearchParams(String(search || "")).get("date") || "";
  } catch {
    return "";
  }
}

function isAppShellSearch(search) {
  try {
    return new URLSearchParams(String(search || "")).get("app") === "1";
  } catch {
    return false;
  }
}

function navigateWithSearch(nextSearch) {
  const current = new URLSearchParams(window.location.search.replace(/^\?/, ""));
  const next = new URLSearchParams(String(nextSearch || "").replace(/^\?/, ""));
  if (current.get("app") === "1") {
    next.set("app", "1");
  }
  const query = next.toString();
  const url = query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
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

function PublicFrame({ headerProps = {}, sideNavProps = {}, children, appShell = false }) {
  if (appShell) {
    return (
      <main className="racing-intel-page racing-intel-page--app-shell">
        <div className="racing-intel-page__shell racing-intel-page__shell--app-shell">
          <div className="racing-intel-page__main racing-intel-page__main--app-shell">
            {children}
          </div>
        </div>
      </main>
    );
  }

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
  const isAppShell = isAppShellSearch(search);
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

  const shouldLoadPublicHeaderData = !isAdminConsole && !isAdminWorkspace && !isAppShell;
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
  const boardRaces = useMemo(
    () => mergeMorningPreviewRaces(races, data?.morning_preview),
    [data?.morning_preview, races],
  );
  const selectedRace = isRaceDetail
    ? resolveSelectedRace(boardRaces, raceDetailId)
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
        appShell={isAppShell}
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
        appShell={isAppShell}
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "reports",
        }}
      >
        <div className="public-content-stack">
          <DailyReportsPage appBasePath={APP_BASE_PATH} appShell={isAppShell} />
        </div>
      </PublicFrame>
    );
  }

  if (isReportDetail) {
    return (
      <PublicFrame
        appShell={isAppShell}
        headerProps={publicHeaderProps}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "reportDetail",
        }}
      >
        <div className="public-content-stack">
          <DailyReportDetailPage slug={reportSlug} appBasePath={APP_BASE_PATH} appShell={isAppShell} />
        </div>
      </PublicFrame>
    );
  }

  if (loading) {
    return (
      <PublicFrame
        appShell={isAppShell}
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
        appShell={isAppShell}
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
        appShell={isAppShell}
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
          {isAppShell ? (
            <section className="app-shell-filter-strip">
              <FilterBar data={data} search={search} onApply={navigateWithSearch} />
            </section>
          ) : null}
          <HistoryPage data={data} appShell={isAppShell} />
        </div>
      </PublicFrame>
    );
  }

  if (isRaceDetail) {
    if (!selectedRace) {
      return (
        <PublicFrame
          appShell={isAppShell}
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
        appShell={isAppShell}
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
          <RaceDetailPage race={selectedRace} search={search} appShell={isAppShell} />
        </div>
      </PublicFrame>
    );
  }

  return (
    <PublicFrame
      appShell={isAppShell}
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
      <div
        className={[
          "public-content-stack",
          isAppShell ? "public-content-stack--app-shell" : "public-content-stack--home",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {!isAppShell && !isDateFocusedHome ? (
          <>
            <MorningPreviewSection data={data} search={search} />
          </>
        ) : null}

        <section
          className={[
            "today-races-section",
            isAppShell ? "today-races-section--app-shell" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          id="home-race-board"
        >
          {isAppShell ? (
            <div className="app-shell-filter-strip">
              <FilterBar data={data} search={search} onApply={navigateWithSearch} />
            </div>
          ) : (
            <>
              <PageSectionHeader
                kicker="公開レース"
                title={targetDateContext?.raceBoardTitle || "対象日の公開レース"}
                subtitle="比較用の導読を確認したあとに、各レースの印、上位候補、結果、定量モデルごとの判断差をレース単位とモデル単位の両方から見比べられます。"
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
                    value: `${boardRaces.length}レース`,
                  },
                ]}
              />
            </>
          )}
          <TodayBoardContent data={data} races={boardRaces} appShell={isAppShell} />
        </section>
        {!isAppShell ? <SecondaryStatsPanel data={data} /> : null}
        {!isAppShell && !isDateFocusedHome ? <HomeHeroSection data={data} search={search} /> : null}
        {!isAppShell && !isDateFocusedHome ? <MethodSummarySection /> : null}
        {!isAppShell && !isDateFocusedHome ? <FeaturedContentSection data={data} /> : null}
        {!isAppShell && !isDateFocusedHome ? <BeginnerGuideSection /> : null}
      </div>
    </PublicFrame>
  );
}

