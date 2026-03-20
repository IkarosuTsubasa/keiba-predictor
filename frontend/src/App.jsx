import React, { useEffect, useMemo, useState } from "react";
import AdminJobsPage from "./components/AdminJobsPage";
import AdminWorkspacePage from "./components/AdminWorkspacePage";
import AppHeader from "./components/AppHeader";
import EmptyRaceState from "./components/EmptyRaceState";
import PageSectionHeader from "./components/PageSectionHeader";
import PublicStaticPage from "./components/PublicStaticPage";
import RaceGrid, { sortRacesForDisplay } from "./components/RaceGrid";
import SecondaryStatsPanel from "./components/SecondaryStatsPanel";
import SiteFooter from "./components/SiteFooter";

const APP_BASE_PATH = "/keiba";
const ADMIN_CONSOLE_PATH = `${APP_BASE_PATH}/console`;
const ADMIN_WORKSPACE_PATH = `${ADMIN_CONSOLE_PATH}/workspace`;
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;

const PUBLIC_PAGE_CONTENT = {
  [`${APP_BASE_PATH}/about`]: {
    title: "このサイトについて",
    lead:
      "いかいもAI競馬は、公開データと独自の予測ロジックをもとに、レースごとの印・買い目・結果回顧を整理して提供する競馬分析サイトです。",
    sections: [
      {
        heading: "提供内容",
        paragraphs: [
          "中央競馬と地方競馬を対象に、各レースの予測ボード、AIごとの印、買い目候補、結果回顧を公開しています。",
          "表示される内容は情報提供を目的としたものであり、投票・購入の実行を代行するものではありません。",
        ],
      },
      {
        heading: "分析の考え方",
        paragraphs: [
          "レース情報、出走表、オッズ、過去成績などをもとに複数の予測系を走らせ、比較しやすい形に整えています。",
          "予測は独自に生成されるもので、結果や利益を保証するものではありません。",
        ],
      },
      {
        heading: "公開ポリシー",
        paragraphs: [
          "予想・印・買い目・回収結果は、公開データと内部処理に基づいて生成していますが、時間差やデータ提供元の変更により表示内容が変わることがあります。",
          "掲載内容は継続的に改善・更新されるため、予告なく構成や表示を変更する場合があります。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/privacy`]: {
    title: "プライバシーポリシー",
    lead:
      "当サイトでは、サービス改善、アクセス解析、広告配信のために、Cookieやログ情報などを利用する場合があります。",
    sections: [
      {
        heading: "1. 取得する情報",
        paragraphs: [
          "アクセス日時、参照元、利用端末、ブラウザ情報、IPアドレス、Cookie等の情報を取得することがあります。",
          "これらの情報は不正利用防止、表示改善、利用状況把握のために使用します。",
        ],
      },
      {
        heading: "2. 広告配信について",
        paragraphs: [
          "当サイトは第三者配信の広告サービスを利用する予定であり、広告配信事業者がCookie等を用いて利用者に適した広告を表示する場合があります。",
          "Google などの第三者配信事業者によるデータ利用については、各事業者のポリシーをご確認ください。",
        ],
      },
      {
        heading: "3. Cookieについて",
        paragraphs: [
          "Cookieは、表示設定の保持、アクセス傾向の把握、広告表示の最適化のために利用される場合があります。",
          "ブラウザ設定によりCookieを無効化することは可能ですが、その場合は一部機能が正しく動作しないことがあります。",
        ],
      },
      {
        heading: "4. 第三者サービスへの情報送信",
        paragraphs: [
          "当サイトでは、広告配信、アクセス解析、ホスティング、セキュリティ対策のために第三者サービスを利用する場合があります。",
          "その際、必要な範囲でアクセス情報や技術情報が外部事業者へ送信されることがあります。",
        ],
      },
      {
        heading: "5. お問い合わせ",
        paragraphs: [
          "個人情報や掲載内容に関するご連絡は、連絡先ページ記載の窓口までお願いいたします。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/terms`]: {
    title: "利用規約・免責事項",
    lead:
      "当サイトの情報は参考情報として提供しており、投資・投票判断の最終決定は利用者自身の責任で行ってください。",
    sections: [
      {
        heading: "1. 適用範囲",
        paragraphs: [
          "本規約は、いかいもAI競馬が提供する公開情報、予想表示、関連ページの利用に関して適用されます。",
          "利用者は、本サイトを利用した時点で本規約に同意したものとみなされます。",
        ],
      },
      {
        heading: "2. サービス内容",
        paragraphs: [
          "当サイトは、AI予想、印、買い目候補、過去結果、各種レース情報を参考情報として公開するサービスです。",
          "掲載情報は情報提供を目的としたものであり、投票勧誘、収益保証、助言契約に該当するものではありません。",
        ],
      },
      {
        heading: "3. 禁止事項",
        paragraphs: [
          "法令または公序良俗に反する行為、当サイトの運営を妨害する行為、他者の権利を侵害する行為を禁止します。",
        ],
      },
      {
        heading: "4. 免責事項",
        paragraphs: [
          "掲載情報の正確性、完全性、最新性については可能な限り注意を払っていますが、保証するものではありません。",
          "当サイトの情報を利用したことにより生じた損害について、運営者は一切の責任を負いません。",
        ],
      },
      {
        heading: "5. 知的財産権",
        paragraphs: [
          "当サイトに掲載される文章、構成、表示、独自集計結果等の知的財産権は、運営者または正当な権利者に帰属します。",
          "引用の範囲を超える転載、再配布、無断利用は禁止します。",
        ],
      },
      {
        heading: "6. 責任ある利用について",
        paragraphs: [
          "当サイトは馬券購入を推奨するものではありません。競馬に関する判断は、法令と節度を守って行ってください。",
          "未成年者は法令に従い、競馬関連サービスを利用しないでください。",
        ],
      },
      {
        heading: "7. 規約の変更",
        paragraphs: [
          "本規約は、必要に応じて事前予告なく変更することがあります。最新の内容は本ページに掲載します。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/disclaimer`]: {
    title: "免責事項",
    lead:
      "当サイトで公開しているAI予想およびレース情報は参考情報であり、最終判断は利用者自身の責任で行ってください。",
    sections: [
      {
        heading: "1. AI予想について",
        paragraphs: [
          "本サイトで公開しているAI予想は、複数モデルによる独自処理の結果であり、結果・的中・収益を保証するものではありません。",
          "AI予想は公開データや処理条件に基づいて生成されるため、実際の結果と異なる場合があります。",
        ],
      },
      {
        heading: "2. 自己責任について",
        paragraphs: [
          "競馬の利用は、ご自身の判断と責任において行ってください。",
          "本サイトの情報を参考にしたことによって生じた損害について、運営者は一切責任を負いません。",
        ],
      },
      {
        heading: "3. 情報の正確性",
        paragraphs: [
          "掲載情報の正確性には配慮していますが、外部データの更新遅延や表示差異により、誤差や遅延が生じる場合があります。",
          "オッズ、払戻金、レース結果などは必ず公式情報をご確認ください。",
        ],
      },
      {
        heading: "4. サービスの中断",
        paragraphs: [
          "保守、障害、外部サービス要因等により、サイトの表示停止、更新遅延、機能制限が発生することがあります。",
        ],
      },
      {
        heading: "5. 外部リンク",
        paragraphs: [
          "当サイトから外部サイトへ移動した場合、リンク先の内容や安全性については責任を負いません。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/contact`]: {
    title: "お問い合わせ",
    lead:
      "掲載内容の修正依頼、権利関係のご連絡、運営に関するご質問は、以下の窓口までお送りください。",
    sections: [
      {
        heading: "連絡先",
        paragraphs: [
          "メール: salvasshaggyya226@gmail.com",
          "通常は数営業日以内の返信を予定しています。内容によっては回答できない場合があります。",
        ],
      },
      {
        heading: "ご連絡時のお願い",
        paragraphs: [
          "対象ページのURL、該当レースID、要件の詳細を明記してください。",
          "権利侵害や掲載削除のご連絡は、本人確認に必要な情報を添えてください。",
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
        <span className="public-screen-state__eyebrow">Loading</span>
        <h1>本日の公開レースを読み込み中です</h1>
        <p>公開ボード API から最新データを取得しています。</p>
      </section>
    </main>
  );
}

function ErrorState({ error, onRetry }) {
  return (
    <main className="public-screen-state">
      <section className="public-screen-state__panel public-screen-state__panel--error">
        <span className="public-screen-state__eyebrow">Error</span>
        <h1>公開情報を表示できません</h1>
        <p>{error || "読み込みに失敗しました。"}</p>
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
  const staticPage = PUBLIC_PAGE_CONTENT[normalizedPath] || null;

  useEffect(() => {
    if (isAdminWorkspace) {
      document.title = "Workspace | いかいもAI競馬";
      return;
    }
    if (isAdminConsole) {
      document.title = "管理コンソール | いかいもAI競馬";
      return;
    }
    if (staticPage) {
      document.title = `${staticPage.title} | いかいもAI競馬`;
      return;
    }
    document.title = "いかいもAI競馬";
  }, [isAdminConsole, isAdminWorkspace, staticPage]);

  const { loading, error, data } = useBoardData(search, !isAdminConsole && !isAdminWorkspace && !staticPage);
  const races = useMemo(() => sortRacesForDisplay(data?.races || []), [data]);

  if (isAdminWorkspace) {
    return <AdminWorkspacePage appBasePath={APP_BASE_PATH} />;
  }

  if (isAdminConsole) {
    return <AdminJobsPage appBasePath={APP_BASE_PATH} />;
  }

  if (staticPage) {
    return (
      <main className="racing-intel-page">
        <AppHeader showFilters={false} />
        <div className="racing-intel-page__shell">
          <PublicStaticPage page={staticPage} />
        </div>
        <SiteFooter />
      </main>
    );
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
      <SiteFooter />
    </main>
  );
}
