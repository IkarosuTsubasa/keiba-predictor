import React, { useEffect, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import BeginnerGuideSection from "./components/BeginnerGuideSection";
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
import { buildTargetDateContext } from "./lib/homepage";
import { matchRaceIdentifier } from "./lib/publicRace";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;
const SITE_NAME = "いかいもAI競馬";
const HOME_PAGE_TITLE = "いかいもAI競馬 | 独自モデルとLLMで競馬予想を比較・検証する分析サイト";

const PUBLIC_PAGE_CONTENT = {
  [`${APP_BASE_PATH}/about`]: {
    kicker: "About",
    title: "このサイトについて",
    lead:
      "いかいもAI競馬は、独自の定量モデルで有力馬を抽出し、その評価をもとに複数のLLMが買い目の考え方を提示する競馬分析サイトです。予想だけを並べるのではなく、公開、比較、結果確認、履歴検証までを同じ導線で見られるように整えています。",
    meta: [
      { label: "サイトの軸", value: "独自モデル / LLM比較 / 公開検証" },
      { label: "公開ページ", value: "予測一覧 / レース詳細 / 履歴分析" },
      { label: "見られるもの", value: "印 / 買い目 / 比較 / 回収率 / 振り返り" },
    ],
    sections: [
      {
        heading: "このサイトがやっていること",
        paragraphs: [
          "このサイトでは、独自の定量モデルで各レースの有力馬と評価順を整理し、その情報をもとに複数のLLMがどのように買い目を組み立てるかを公開しています。",
          "予想を出して終わりではなく、レースごとの結果、回収率、履歴分析まで同じ流れで確認できるようにし、モデルごとの特徴や差を追いやすくしています。",
        ],
      },
      {
        heading: "定量モデルとLLMの関係",
        paragraphs: [
          "定量モデルは馬券を直接出すための装置ではなく、各馬の評価順や有力候補を整理するための分析軸です。どの馬にどれだけ注目が集まっているかを測る土台として使っています。",
          "LLMはその評価を受けて、券種の選び方、点数の置き方、見送り判断の違いを提示します。つまり、定量モデルが土台をつくり、LLMが買い目構成の差を見せる役割を担います。",
        ],
      },
      {
        heading: "このサイトの見方",
        paragraphs: [
          "まずはトップで対象日の見どころと公開レース全体を確認し、気になるレースがあればレース詳細でAIモデル別の買い目と定量モデルの本命比較を見るのが基本的な使い方です。",
          "そのうえで履歴分析まで遡ると、単日の当たり外れだけでは見えにくいモデルごとの傾向や再現性を把握しやすくなります。公開情報は参考情報であり、最終的な判断はご自身で行ってください。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/guide`]: {
    kicker: "Guide",
    title: "ガイド",
    lead:
      "初めて見る場合は、トップ、レース詳細、履歴分析の順で確認すると全体像を掴みやすくなります。ここでは各ページで何を見て、どこを比較するとサイトの特徴が分かるかを簡潔にまとめています。",
    meta: [
      { label: "最初に見る", value: "トップ" },
      { label: "次に見る", value: "レース詳細" },
      { label: "最後に比べる", value: "履歴分析" },
    ],
    sections: [
      {
        heading: "トップの見方",
        paragraphs: [
          "トップでは、まず対象日の見どころ、分析方法、注目コンテンツを確認し、そのあとで公開レースの一覧に進む構成になっています。",
          "データだけを見るのではなく、どのレースを先に読むべきか、どこで意見が割れているか、どのような比較軸で見るべきかをつかんでからレース一覧を見るのが基本です。",
        ],
      },
      {
        heading: "レース詳細の見方",
        paragraphs: [
          "レース詳細では、AIモデル別の買い目、コメント、結果を同じ画面で確認できます。どのモデルがどの券種を選び、何点で組み、どういう結果になったかを一つずつ追えます。",
          "あわせて定量モデルの本命比較を見ると、買い目の方向性と評価順の関係、どこでズレが出ているかを確認できます。モデル一致度を見る場としてもレース詳細が中心になります。",
        ],
      },
      {
        heading: "履歴分析の見方",
        paragraphs: [
          "履歴分析では、AIモデルの回収成績と定量モデルの傾向を、月間・年間・累計で見比べられます。単日の結果ではなく、期間で見たときにどのモデルがどの条件で強いかを把握するためのページです。",
          "トップやレース詳細で感じた印象を、その日だけの偶然で終わらせず、履歴で確かめに行く使い方を想定しています。",
        ],
      },
      {
        heading: "どこを比較すればいいか",
        paragraphs: [
          "まずは本命が揃っているか、券種が揃っているか、見送り判断に差があるかの3点を見ると、各モデルの考え方の違いがつかみやすくなります。",
          "そのうえで結果や履歴分析まで追うと、単なる一発の当たり外れではなく、モデルごとの得意不得意やリスクの置き方まで読み取りやすくなります。",
        ],
      },
      {
        heading: "対象日の切り替え",
        paragraphs: [
          "左側の対象日から日付を指定できます。前日・翌日の切り替えボタンでも公開日をすばやく移動できます。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/methodology`]: {
    kicker: "Methodology",
    title: "分析方針",
    lead:
      "いかいもAI競馬では、独自の定量モデルで有力馬を抽出し、その評価をもとに複数のLLMが買い目を組み立てます。重要なのは、どれか一つの答えを絶対視することではなく、どの視点がどこで揃い、どこで割れ、結果としてどうだったかを同じ形式で追い続けることです。",
    meta: [
      { label: "Step 1", value: "定量モデルで有力馬を抽出" },
      { label: "Step 2", value: "LLMで買い目構成を比較" },
      { label: "Step 3", value: "結果と履歴で継続検証" },
    ],
    sections: [
      {
        heading: "独自モデルで有力馬を抽出する",
        paragraphs: [
          "最初の土台になるのは定量モデルです。過去成績、条件適性、位置取り、想定オッズとのバランスなどを踏まえて、有力馬と評価順の輪郭を整理します。",
          "ここで重視しているのは、単に順位を出すことではなく、どの馬が軸になりやすいか、どこから評価が落ちるか、どのレースが素直でどのレースが難解かを見極めることです。",
        ],
      },
      {
        heading: "LLMで買い目構成を比較する",
        paragraphs: [
          "LLMは定量モデルの評価やレース条件を受け取り、どの券種を選ぶか、何点に絞るか、資金配分をどう考えるかといった買い目構成を提示します。",
          "同じ有力馬を見ていても、券種の選び方や見送り判断はモデルごとに変わります。この差を見比べることで、単なる印の一致だけでは分からない思想の違いが見えてきます。",
        ],
      },
      {
        heading: "公開、結果、履歴で検証する",
        paragraphs: [
          "本サイトでは、公開した予想をその場限りで終わらせません。結果、回収率、期間別の成績まで同じ導線で確認できるようにし、単日の印象と履歴の実績を行き来できる構造にしています。",
          "重視しているのは、短期の当たり外れだけでなく、どのモデルがどんな条件で強いか、どこで慎重に見るべきかを継続して検証することです。公開情報は参考情報であり、最終的な判断は利用者自身で行ってください。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/privacy`]: {
    title: "プライバシーポリシー",
    lead:
      "本サイトでは、表示改善や障害調査のために必要な範囲で一般的なアクセス情報を取り扱う場合があります。",
    sections: [
      {
        heading: "取得する情報",
        paragraphs: [
          "アクセス日時、利用端末に関する基本情報、表示改善のために必要な技術情報など、一般的なアクセス解析に準じた範囲の情報を取り扱う場合があります。",
        ],
      },
      {
        heading: "利用目的",
        paragraphs: [
          "表示改善、障害調査、不正利用対策など、サイト運営に必要な目的に限って利用します。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/terms`]: {
    title: "利用規約",
    lead:
      "本サイトを利用する際は、表示内容が参考情報であることをご理解のうえご利用ください。利用は自己責任で行うものとします。",
    sections: [
      {
        heading: "利用上の注意",
        paragraphs: [
          "公開情報の正確性や継続性について保証するものではありません。表示内容は予告なく変更される場合があります。",
        ],
      },
      {
        heading: "禁止事項",
        paragraphs: [
          "不正アクセス、運営の妨害、法令や公序良俗に反する利用は禁止します。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/disclaimer`]: {
    title: "免責事項",
    lead:
      "本サイトの情報は、投資判断や馬券購入を保証するものではありません。利用により生じたいかなる結果についても、運営側は責任を負いません。",
    sections: [
      {
        heading: "参考情報について",
        paragraphs: [
          "公開している買い目、結果、履歴分析は参考情報です。最終的な判断は利用者ご自身で行ってください。",
        ],
      },
      {
        heading: "表示の変動について",
        paragraphs: [
          "対象レース数、公開タイミング、履歴集計の条件によって数値は変動します。単日の結果だけで評価しないことを推奨します。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/contact`]: {
    title: "お問い合わせ",
    lead:
      "表示内容や公開ページに関するお問い合わせは、下記の連絡先までお願いします。確認には時間がかかる場合があります。",
    sections: [
      {
        heading: "連絡先",
        paragraphs: [
          "メール: salvasshaggyya226@gmail.com",
          "お問い合わせの際は、対象ページの URL や発生状況をあわせて記載してください。",
        ],
      },
    ],
  },
};

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
    <main className="public-screen-state">
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
    </main>
  );
}

function ErrorState({ error, onRetry }) {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel public-screen-state__panel--error">
        <span className="public-screen-state__eyebrow">エラー</span>
        <h1>公開情報を表示できませんでした</h1>
        <p>{error || "読み込みに失敗しました。"}</p>
        <button type="button" onClick={onRetry}>
          再読み込み
        </button>
      </section>
    </main>
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
  const raceDetailId = extractRaceDetailId(normalizedPath);
  const isRaceDetail = Boolean(raceDetailId);
  const staticPage = PUBLIC_PAGE_CONTENT[normalizedPath] || null;
  const selectedDate = extractSelectedDate(search);
  const isDateFocusedHome = normalizedPath === APP_BASE_PATH && Boolean(selectedDate);
  useEffect(() => {
    if (isAdminWorkspace) {
      document.title = `Workspace | ${SITE_NAME}`;
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
    if (staticPage) {
      document.title = `${staticPage.title} | ${SITE_NAME}`;
      return;
    }
    document.title = HOME_PAGE_TITLE;
  }, [isAdminConsole, isAdminWorkspace, isHistoryPage, isRaceDetail, staticPage]);

  const { loading, error, data } = useBoardData(
    search,
    !isAdminConsole && !isAdminWorkspace && !staticPage,
  );
  const targetDateContext = data ? buildTargetDateContext(data) : null;
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
        headerProps={{ showFilters: false }}
        sideNavProps={{ pathname: normalizedPath, mode: "static" }}
      >
        <div className="public-content-stack">
          <PublicStaticPage page={staticPage} />
        </div>
      </PublicFrame>
    );
  }

  if (loading) {
    return <LoadingState />;
  }

  if (error || !data) {
    return (
      <ErrorState
        error={error}
        onRetry={() => setSearch(window.location.search.replace(/^\?/, ""))}
      />
    );
  }

  if (isHistoryPage) {
    return (
      <PublicFrame
        headerProps={{ showFilters: false }}
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
          headerProps={{ showFilters: false }}
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
        headerProps={{ showFilters: false }}
        sideNavProps={{
          pathname: normalizedPath,
          mode: "detail",
          detailHref: `${normalizedPath}${buildQuery(search)}`,
          detailTitle: selectedRace?.display_header?.title || "レース詳細",
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
      headerProps={{ showFilters: false }}
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
            kicker="Race Board"
            title={targetDateContext?.raceBoardTitle || "対象日の公開レース"}
            subtitle="比較用の導読を確認したあとに、各レースの印、買い目、結果、回収率をレース単位とモデル単位の両方から見比べられます。"
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

