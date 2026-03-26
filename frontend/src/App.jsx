import React, { useEffect, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import HeroSpotlightStrip from "./components/HeroSpotlightStrip";
import HistoryPage from "./components/HistoryPage";
import PageSectionHeader from "./components/PageSectionHeader";
import PublicSideNav from "./components/PublicSideNav";
import PublicStaticPage from "./components/PublicStaticPage";
import RaceDetailPage from "./components/RaceDetailPage";
import SecondaryStatsPanel from "./components/SecondaryStatsPanel";
import SiteFooter from "./components/SiteFooter";
import SocialBarLoader from "./components/SocialBarLoader";
import TodayBoardContent from "./components/TodayBoardContent";
import { matchRaceIdentifier } from "./lib/publicRace";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;
const SITE_NAME = "いかいもAI競馬";

const PUBLIC_PAGE_CONTENT = {
  [`${APP_BASE_PATH}/about`]: {
    kicker: "About",
    title: "このサイトについて",
    lead:
      "いかいもAI競馬は、公開レースの予測一覧、単場詳細、履歴分析を同じ導線で見比べられる競馬予測サイトです。AI モデルの印と買い目、量化モデルの比較、公開後の成績推移を一つの画面設計で横断できるように整えています。",
    meta: [
      { label: "公開ページ", value: "予測一覧 / 単場詳細 / 履歴分析" },
      { label: "AI モデル", value: "印・買い目・回収率を表示" },
      { label: "量化モデル", value: "本命比較と期間成績を表示" },
    ],
    sections: [
      {
        heading: "公開ビューの役割",
        paragraphs: [
          "トップでは当日の公開レースを一覧で確認し、各レースごとの印、買い目、結果、回収率を同じ流れで見られます。",
          "単場詳細では AI モデルの買い目と、量化モデルの本命比較やコンセンサスを一つの画面で把握できます。",
        ],
      },
      {
        heading: "AI モデルと量化モデル",
        paragraphs: [
          "AI モデルは公開中の買い目と印を表示するためのモデル群です。実際の公開買い目や回収率はこの AI モデル側で確認できます。",
          "量化モデルは馬券購入のためではなく、各馬の評価や並び順を比較するための分析モデルです。履歴分析では期間ごとの強みを見比べられます。",
        ],
      },
      {
        heading: "見方の考え方",
        paragraphs: [
          "単日の当たり外れだけでなく、履歴分析で月間・年間・累計の傾向を見ながら、どのモデルがどの条件で強いかを確認する使い方を想定しています。",
          "公開情報は参考情報であり、最終的な判断はご自身で行ってください。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/guide`]: {
    kicker: "Guide",
    title: "ガイド",
    lead:
      "初めて見る場合は、トップ、単場詳細、履歴分析の順で確認すると全体像を掴みやすくなります。ここでは各ページで何を見るべきかを簡潔にまとめています。",
    meta: [
      { label: "最初に見る", value: "トップ" },
      { label: "詳しく見る", value: "単場詳細" },
      { label: "比較する", value: "履歴分析" },
    ],
    sections: [
      {
        heading: "トップの見方",
        paragraphs: [
          "トップでは公開レースの一覧を見ながら、各モデルの本命、買い目、ROI を横並びで確認できます。",
          "まずはどのレースが公開されているか、どのモデルが本命をどこに置いているかを見るのが基本です。",
        ],
      },
      {
        heading: "単場詳細の見方",
        paragraphs: [
          "単場詳細では AI モデル別の買い目をまとめて確認できます。どのモデルが何件の買い目を出しているか、結果がどうだったかを一つずつ追えます。",
          "あわせて量化モデルの本命比較を見ることで、公開買い目と量化評価のズレも確認できます。",
        ],
      },
      {
        heading: "履歴分析の見方",
        paragraphs: [
          "履歴分析では AI モデルの回収成績と量化モデルの的中傾向を、月間・年間・累計で見比べられます。",
          "単日の結果ではなく、期間ごとの安定感や得意な傾向を見るためのページです。",
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
      "数字は、偶然を物語に変えるためではなく、物語の奥に潜む秩序を照らし出すためにあります。いかいもAI競馬では、公開予測をただ並べるのではなく、AI と量化モデルの視点を重ねることで、レースの輪郭をより立体的に浮かび上がらせることを目指しています。",
    meta: [
      { label: "AI モデル", value: "公開買い目の判断軸" },
      { label: "量化モデル", value: "評価順の比較軸" },
      { label: "履歴分析", value: "期間で見る強みと癖" },
    ],
    sections: [
      {
        heading: "一つの正解より、複数の視点",
        paragraphs: [
          "競馬の予測は、単一の答えを探す作業ではありません。展開、能力、適性、市場評価、そのどれもがレースの表情を変えます。",
          "だからこそ本サイトでは、AI モデルの公開買い目と量化モデルの比較を分けて提示し、それぞれの視点がどこで重なり、どこでズレるのかを見える形にしています。",
        ],
      },
      {
        heading: "公開情報の扱い方",
        paragraphs: [
          "公開中の買い目や結果は、単日の当たり外れだけで評価するためのものではありません。履歴分析まで含めて眺めることで、モデルごとの癖や得意な局面が見えてきます。",
          "数字の上下に一喜一憂するのではなく、期間の流れと再現性を見に行くことを、このサイトでは重視しています。",
        ],
      },
      {
        heading: "量化モデルの立ち位置",
        paragraphs: [
          "量化モデルは馬券を直接出すための装置ではなく、各馬の評価と並び順を比較するための軸です。公開買い目の背景にある温度差を測るための、もう一つの物差しとして置いています。",
          "本命一致や上位馬の馬券内率は、派手な演出ではなく、モデルの輪郭を静かに語るための指標です。",
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
        <h1>本日の公開レースを読み込んでいます</h1>
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

function PublicFrame({
  headerProps = {},
  sideNavProps = {},
  showSocialBar = false,
  children,
}) {
  const shouldShowSocialBar = Boolean(
    showSocialBar || sideNavProps.mode === "home" || sideNavProps.mode === "detail",
  );

  return (
    <main className="racing-intel-page">
      <AppHeader {...headerProps} />
      <div className="racing-intel-page__shell racing-intel-page__shell--with-sidebar">
        <PublicSideNav {...sideNavProps} />
        <div className="racing-intel-page__main">{children}</div>
      </div>
      <SiteFooter />
      <SocialBarLoader enabled={shouldShowSocialBar} />
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
  const showSocialBar = normalizedPath === APP_BASE_PATH || isRaceDetail;

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
    document.title = SITE_NAME;
  }, [isAdminConsole, isAdminWorkspace, isHistoryPage, isRaceDetail, staticPage]);

  const { loading, error, data } = useBoardData(
    search,
    !isAdminConsole && !isAdminWorkspace && !staticPage,
  );
  const races = data?.races || [];
  const selectedRace = isRaceDetail
    ? races.find((race) => matchRaceIdentifier(race, raceDetailId))
    : null;

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
        sideNavProps={{ pathname: normalizedPath, mode: "history" }}
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
          }}
          showSocialBar={showSocialBar}
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
      <div className="public-content-stack">
        <section className="today-races-section">
          <PageSectionHeader
            kicker=""
            title={selectedDate ? "レースAI予測" : "本日のAI予測"}
            subtitle="複数モデルの印、買い目案、結果、回収率をレース単位とモデル単位の両方から確認できます。"
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
              {
                key: "generated-at",
                label: "最終更新",
                value: data?.generated_at_label || "-",
              },
            ]}
          />

          <HeroSpotlightStrip data={data} />
          <TodayBoardContent data={data} races={races} />
        </section>

        <SecondaryStatsPanel data={data} />
      </div>
    </PublicFrame>
  );
}
