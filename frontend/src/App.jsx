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
import TodayBoardContent from "./components/TodayBoardContent";
import { matchRaceIdentifier } from "./lib/publicRace";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;
const SITE_NAME = "競馬AIインテリジェンス";

const PUBLIC_PAGE_CONTENT = {
  [`${APP_BASE_PATH}/about`]: {
    title: "このサイトについて",
    lead:
      "競馬AIインテリジェンスは、レースごとの公開予測、モデル別の傾向、回収率の推移を見やすく整理して公開するための情報ページです。投票判断の前に、モデルの特徴と数字の背景を落ち着いて確認できる構成を目指しています。",
    sections: [
      {
        heading: "公開方針",
        paragraphs: [
          "このサイトでは、当日の公開レースを中心に、複数モデルの印、買い目案、回収率などをまとめて表示します。情報量は保ちつつも、視認性を優先した整理を行い、必要な数字へ素早く辿り着ける画面を目指しています。",
          "派手な演出よりも、比較しやすさと読みやすさを重視しています。競馬の判断材料を一枚の画面で俯瞰できることを最優先にしています。",
        ],
      },
      {
        heading: "対象ユーザー",
        paragraphs: [
          "単発の印だけではなく、モデルの継続成績や回収率の偏りまで確認したい方を想定しています。数字の推移やレースごとの差分を見ながら、自分の判断材料として使いたい方向けの設計です。",
        ],
      },
      {
        heading: "ご利用にあたって",
        paragraphs: [
          "掲載情報は参考情報であり、最終的な投票判断は利用者ご自身の責任でお願いします。モデルの結果には変動があり、将来の的中や回収を保証するものではありません。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/guide`]: {
    title: "ガイド",
    lead:
      "トップページでは、当日の注目指標、レース別表示、モデル別表示、通算成績を確認できます。初めて利用する場合は、まずレース別表示から印と買い目案の見方を確認してください。",
    sections: [
      {
        heading: "レース別表示",
        paragraphs: [
          "各レースカードには、モデルごとの印、買い目案、結果、回収率がまとまっています。まずはレース単位で比較したい場合に適しています。",
        ],
      },
      {
        heading: "モデル別表示",
        paragraphs: [
          "モデル別表示では、選択したモデルの対象レースだけを一覧できます。モデル単位で得意傾向や当日の精度を見たい場合に便利です。",
        ],
      },
      {
        heading: "指標の見方",
        paragraphs: [
          "ROI は回収率、的中は結果が条件に合致した件数、掲載数は公開対象になったレース数を示します。数字だけでなく、対象件数も合わせて確認してください。",
        ],
      },
      {
        heading: "日付切替",
        paragraphs: [
          "ヘッダーの日付入力から公開対象日を切り替えられます。日によって公開レース数や集計結果が大きく変わるため、更新時刻も合わせて確認するのがおすすめです。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/methodology`]: {
    title: "分析方針",
    lead:
      "公開画面では、モデルの強弱を単純な印だけで見せるのではなく、結果・回収率・買い目案のまとまりとして提示しています。数字を過剰に飾らず、比較しやすい形で整理することを基本方針としています。",
    sections: [
      {
        heading: "表示設計",
        paragraphs: [
          "重要な数値は大きく、補助情報は控えめに表示し、画面全体に自然な優先順位を作っています。レースカードごとの密度は維持しつつ、読み疲れしない余白を確保しています。",
        ],
      },
      {
        heading: "モデル比較",
        paragraphs: [
          "同一レース内でモデルの印と買い目案を比較できるようにし、さらにモデル別表示では横断的な確認ができる構成にしています。単発の結果だけでなく、当日の並び全体でモデルの癖を掴めるようにしています。",
        ],
      },
      {
        heading: "数値の扱い",
        paragraphs: [
          "的中率や ROI は参考指標であり、条件数や公開件数と合わせて見ることを前提にしています。短期間の数字だけで結論を出さず、継続的な傾向の確認を重視しています。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/privacy`]: {
    title: "プライバシーポリシー",
    lead:
      "本サイトでは、利用状況の把握や表示改善のために、一般的なアクセス情報を取得する場合があります。取得した情報は、運営と分析の目的に限って取り扱います。",
    sections: [
      {
        heading: "取得する情報",
        paragraphs: [
          "アクセス日時、参照元、利用端末に関する基本的な技術情報など、一般的なアクセス解析に必要な範囲の情報を取得することがあります。",
        ],
      },
      {
        heading: "利用目的",
        paragraphs: [
          "サイト表示の改善、障害調査、不正利用の検知、公開情報の品質向上のために利用します。個人を特定する目的での利用は行いません。",
        ],
      },
      {
        heading: "外部サービス",
        paragraphs: [
          "アクセス解析や広告配信など、必要に応じて外部サービスを利用する場合があります。各サービスの詳細は、その提供元の方針をご確認ください。",
        ],
      },
      {
        heading: "お問い合わせ",
        paragraphs: [
          "個人情報や運用に関するお問い合わせは、連絡先ページからお送りください。確認のうえ、必要な範囲で対応します。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/terms`]: {
    title: "利用規約",
    lead:
      "本サイトの利用にあたっては、以下の内容をご確認ください。公開情報は参考情報として提供されており、利用者は自身の判断と責任で利用するものとします。",
    sections: [
      {
        heading: "サービス内容",
        paragraphs: [
          "本サイトは、競馬に関する公開予測、集計結果、回収率指標などを提供します。内容は予告なく変更、更新、停止される場合があります。",
        ],
      },
      {
        heading: "禁止事項",
        paragraphs: [
          "法令または公序良俗に反する行為、運営を妨害する行為、掲載データの不正利用、他者に損害を与える行為を禁止します。",
        ],
      },
      {
        heading: "知的財産",
        paragraphs: [
          "本サイトに掲載される文章、構成、デザイン、データ整理の方法には運営者または権利者の権利が帰属します。無断転載や再配布はご遠慮ください。",
        ],
      },
      {
        heading: "変更と停止",
        paragraphs: [
          "保守や障害対応、運営上の判断により、サービス内容を変更または停止することがあります。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/disclaimer`]: {
    title: "免責事項",
    lead:
      "本サイトの情報は、正確性と見やすさに配慮して掲載していますが、その完全性や将来の成績を保証するものではありません。利用は自己責任でお願いします。",
    sections: [
      {
        heading: "予測情報について",
        paragraphs: [
          "公開される印、買い目案、回収率、モデル比較は参考情報です。実際の投票判断に用いる際は、利用者自身で内容を確認のうえご判断ください。",
        ],
      },
      {
        heading: "成績の変動",
        paragraphs: [
          "モデル成績は対象レースや期間によって変動します。過去の数値が将来の結果を保証するものではありません。",
        ],
      },
      {
        heading: "損害等",
        paragraphs: [
          "本サイトの情報を利用したことによって生じた損失、損害、機会損失などについて、運営者は責任を負いません。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/contact`]: {
    title: "お問い合わせ",
    lead:
      "表示内容や公開方針、掲載データに関するお問い合わせは、以下の連絡先までお願いします。内容を確認のうえ、必要な範囲で回答します。",
    sections: [
      {
        heading: "連絡先",
        paragraphs: [
          "メール: salvasshaggyya226@gmail.com",
          "返信には時間をいただく場合があります。あらかじめご了承ください。",
        ],
      },
      {
        heading: "お問い合わせ時のお願い",
        paragraphs: [
          "対象ページの URL、確認したレース、発生している内容をできるだけ具体的に記載してください。",
          "表示崩れや文言の不備については、利用端末や閲覧日時を添えていただけると確認がスムーズです。",
        ],
      },
    ],
  },
};

function buildQuery(search) {
  return search ? `?${search}` : "";
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

  useEffect(() => {
    if (isAdminWorkspace) {
      document.title = `Workspace | ${SITE_NAME}`;
      return;
    }
    if (isAdminConsole) {
      document.title = `運用コンソール | ${SITE_NAME}`;
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
        <div className="racing-intel-page__shell">
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
        <div className="racing-intel-page__shell">
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
        >
          <div className="racing-intel-page__shell">
            <section className="empty-race-state">
              <span className="empty-race-state__eyebrow">レース詳細</span>
              <h2>該当するレースが見つかりません</h2>
              <p>
                一覧ページに戻って、日付や公開範囲を切り替えてから再度お試しください。
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
        <div className="racing-intel-page__shell">
          <RaceDetailPage race={selectedRace} search={search} />
        </div>
      </PublicFrame>
    );
  }

  return (
    <PublicFrame
      headerProps={{
        data,
        search,
        onApplyFilters: navigateWithSearch,
      }}
      sideNavProps={{ pathname: normalizedPath, mode: "home" }}
    >
      <div className="racing-intel-page__shell">
        <section className="today-races-section">
          <PageSectionHeader
            kicker="本日の公開レース"
            title="本日のAI予測"
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
