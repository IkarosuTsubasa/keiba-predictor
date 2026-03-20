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
      "いかいもAI競馬は、単一の直感や単一モデルに依存しない競馬分析サイトです。複数の評価軸を重ねながら、公開情報を整理し、比較し、判断材料として読み解ける形に整えて公開しています。",
    sections: [
      {
        heading: "目指していること",
        paragraphs: [
          "このサイトが重視しているのは、一発で当てる印象よりも、複数の視点を並べて比較できる状態をつくることです。人気、オッズ、予想印、買い目、回収結果を切り離さず、ひとつの判断材料として読み取れるよう構成しています。",
          "競馬の予想には、スピード、展開、人気、妙味、買い方の設計など、異なる判断軸が同時に存在します。当サイトではそれらを一層ずつ整理し、見える形で提示することを大切にしています。",
        ],
      },
      {
        heading: "単なるAI予想ではない理由",
        paragraphs: [
          "表示される結論は、単純な一問一答の出力ではありません。複数のモデル観点と公開情報の比較を通じて、どの馬に注目が集まり、どこで評価が割れ、どの券種が見合うのかを段階的に整えています。",
          "そのため、同じレースでも『強く買う』『絞る』『広げる』『見送る』といった判断差が現れます。この差分そのものが、ひとつの重要な分析情報だと考えています。",
        ],
      },
      {
        heading: "公開している価値",
        paragraphs: [
          "当サイトでは、予想そのものだけでなく、予想に至る構造と結果の振り返りも重視しています。各AIの見解、買い目の構成、回収率の推移を同時に見られることで、単発の印象に流されにくい観察が可能になります。",
          "日々のレースを通じて、判断の重なりと差分、そして結果との距離を継続的に検証できることが、このサイトの中核です。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/guide`]: {
    title: "ガイド",
    lead:
      "このページでは、サイト内で表示される印、ROI、買い目、見送り判断などの見方をまとめています。初めて利用する方が、各表示の意味を迷わず理解できるように整理した案内ページです。",
    sections: [
      {
        heading: "このサイトで見られるもの",
        paragraphs: [
          "各レースページでは、複数AIの印、買い目、回収率、公開後の結果が一覧できます。ひとつの答えを押しつけるのではなく、複数の視点を横並びで比較できることが特徴です。",
        ],
      },
      {
        heading: "印の意味",
        paragraphs: [
          "◎は中心視、○は対抗評価、▲は有力な次点、△と☆は押さえや注意枠として扱われます。ただし印は単独で読むのではなく、オッズや買い目の構成と合わせて見ることで意味が深まります。",
        ],
      },
      {
        heading: "複数AIを並べて見る理由",
        paragraphs: [
          "競馬では、同じレースでも重視する要素によって評価が分かれます。複数AIを並べることで、評価が集中している馬と、意見が割れている馬を同時に確認できます。",
          "一致は強さの参考になり、差分は不確実性や妙味のヒントになります。",
        ],
      },
      {
        heading: "ROIと回収率",
        paragraphs: [
          "ROIは投じた金額に対して、どれだけ回収できたかを示す指標です。高い数値だけを見るのではなく、サンプル数、買い目の広さ、的中頻度と合わせて読むことが重要です。",
        ],
      },
      {
        heading: "買い目が出る時と出ない時",
        paragraphs: [
          "当サイトでは、常に買い目を出すことを目的にしていません。評価の裏付けが弱い場合や、妙味とリスクのバランスが合わない場合は、見送り判断もそのまま表示します。",
          "見送りは消極策ではなく、判断の質を整えるための結果として扱っています。",
        ],
      },
      {
        heading: "オッズと妙味",
        paragraphs: [
          "支持されている馬がそのまま良い買い目になるとは限りません。当サイトでは、評価の強さだけでなく、オッズとのバランスを見ながら買い目の構成を読み取れるようにしています。",
        ],
      },
      {
        heading: "予想の受け取り方",
        paragraphs: [
          "本サイトの表示は、最終判断を代行するものではなく、判断材料を整理して提示するものです。複数AIの一致点と差分を見ながら、自分の基準で読む使い方を想定しています。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/methodology`]: {
    title: "分析方法",
    lead:
      "いかいもAI競馬では、ひとつの答えを断定的に提示するのではなく、複数の評価軸を重ねながら、判断の強さと不確実性を同時に読み取れる形で公開しています。このページでは、その考え方の骨格を紹介します。",
    sections: [
      {
        heading: "多面的にレースを見る",
        paragraphs: [
          "競馬の判断は、単一の能力比較だけで完結しません。人気、オッズ、展開想定、印の集中度、買い目の組み方など、複数の視点を並べて初めて全体像が見えてきます。",
          "当サイトでは、個々の要素を一度分解し、それぞれの強弱や重なりを確認できる構成を重視しています。",
        ],
      },
      {
        heading: "一致と差分の両方を評価する",
        paragraphs: [
          "複数のAIが同じ方向を向いている場合、それは判断の強さを測るひとつの手掛かりになります。一方で、見解が割れる場合には、レースに含まれる不確実性や論点のズレが見えてきます。",
          "当サイトでは、一致だけでなく差分も情報として扱い、単純な多数決ではない読み方を可能にしています。",
        ],
      },
      {
        heading: "オッズと評価のバランスを見る",
        paragraphs: [
          "高く評価される馬が、そのまま投票妙味のある対象になるとは限りません。支持と価格がどのように釣り合っているかを見ることで、強さと妙味の距離を把握しやすくなります。",
          "そのため当サイトでは、評価だけでなく、どの券種でどの程度の広さを持たせるかという構成面も重要な判断材料としています。",
        ],
      },
      {
        heading: "見送りも分析結果のひとつ",
        paragraphs: [
          "どのレースでも無理に買い目を出すことは、分析の質を下げる場合があります。評価の裏付けが弱い時、価格とリスクが見合わない時、判断は『見送り』という形で表現されます。",
          "当サイトでは、出す結論の多さではなく、出すべき根拠があるかどうかを重視しています。",
        ],
      },
      {
        heading: "継続的に結果を振り返る",
        paragraphs: [
          "分析の価値は、提示した瞬間だけでは決まりません。公開後の結果、回収率、券種ごとの傾向を継続的に確認することで、判断の癖や再現性が見えてきます。",
          "いかいもAI競馬は、単発の印象に依存するのではなく、継続観測によって判断を磨いていくための競馬分析サイトです。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/privacy`]: {
    title: "プライバシーポリシー",
    lead:
      "当サイトでは、利用状況の把握、表示改善、広告配信の最適化のために、アクセス情報やCookie等の技術情報を利用する場合があります。本ページでは、その取扱い方針を説明します。",
    sections: [
      {
        heading: "1. 取得する情報",
        paragraphs: [
          "当サイトでは、アクセス日時、閲覧ページ、参照元、ブラウザ種別、端末情報、IPアドレス、Cookie等の情報を取得する場合があります。",
          "これらはサイト運営、表示最適化、不正利用対策、広告配信の改善に利用されます。",
        ],
      },
      {
        heading: "2. 利用目的",
        paragraphs: [
          "取得した情報は、サイト品質の維持、閲覧傾向の分析、表示内容の改善、問い合わせ対応、広告の最適化のために利用します。",
          "個人を特定する目的で第三者に販売することはありません。",
        ],
      },
      {
        heading: "3. Cookieについて",
        paragraphs: [
          "当サイトでは、利便性向上や広告配信のためにCookieを利用する場合があります。Cookieにより、再訪時の表示調整や利用傾向の把握が行われることがあります。",
          "ブラウザ設定によりCookieを無効化することは可能ですが、一部機能や表示が正しく動作しない場合があります。",
        ],
      },
      {
        heading: "4. 広告配信について",
        paragraphs: [
          "当サイトは、第三者配信の広告サービスを利用する場合があります。広告配信事業者は、利用者の関心に応じた広告表示のためにCookie等を使用することがあります。",
          "Googleを含む第三者事業者によるデータ利用については、各社のポリシーをご確認ください。",
        ],
      },
      {
        heading: "5. お問い合わせ",
        paragraphs: [
          "個人情報や本ポリシーに関するご連絡は、Contactページ記載のメールアドレスまでお願いします。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/terms`]: {
    title: "利用規約",
    lead:
      "本規約は、いかいもAI競馬が提供する情報、表示、分析コンテンツの利用条件を定めるものです。ご利用の前に内容をご確認ください。",
    sections: [
      {
        heading: "1. サービスの性質",
        paragraphs: [
          "本サイトは、競馬に関する公開情報、分析情報、AI予想表示、買い目参考情報を提供する情報サービスです。投票の受付や資金管理を行うものではありません。",
        ],
      },
      {
        heading: "2. 利用者の責任",
        paragraphs: [
          "本サイトの情報は、利用者ご自身の判断材料としてご利用ください。実際の投票、資金配分、行動判断は、すべて利用者ご本人の責任で行うものとします。",
        ],
      },
      {
        heading: "3. 表示内容について",
        paragraphs: [
          "当サイトは、情報の正確性、完全性、継続性の確保に努めますが、内容の完全な正確性や将来結果との一致を保証するものではありません。",
          "表示される予想、印、買い目、回収率は参考情報であり、利益を保証するものではありません。",
        ],
      },
      {
        heading: "4. 禁止事項",
        paragraphs: [
          "法令違反、公序良俗違反、サイト運営の妨害、不正アクセス、情報の無断転載、誤認を招く形での再配布その他当サイトが不適切と判断する行為を禁止します。",
        ],
      },
      {
        heading: "5. 変更と停止",
        paragraphs: [
          "当サイトは、予告なく内容の変更、更新、一時停止または終了を行うことがあります。",
        ],
      },
      {
        heading: "6. 免責",
        paragraphs: [
          "当サイトの利用により生じた損害について、運営者は可能な範囲を除き責任を負いません。利用者は自らの判断でご利用ください。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/disclaimer`]: {
    title: "免責事項",
    lead:
      "当サイトで公開しているAI予想、買い目、指標、結果整理は、情報提供を目的とした参考コンテンツです。投票判断や金銭的意思決定は、利用者ご自身の責任で行ってください。",
    sections: [
      {
        heading: "1. 予想情報について",
        paragraphs: [
          "当サイトの予想は、複数の分析観点を整理した参考情報です。結果的な的中や回収を保証するものではありません。",
          "過去実績、回収率、表示順位は将来の成果を保証するものではありません。",
        ],
      },
      {
        heading: "2. 投票判断について",
        paragraphs: [
          "実際の投票、金額設定、券種選択は利用者ご自身の判断で行ってください。当サイトは投票判断を代行するものではありません。",
        ],
      },
      {
        heading: "3. 情報の正確性",
        paragraphs: [
          "掲載情報は公開情報に基づき整理していますが、更新タイミングや外部要因により差異が生じる場合があります。当サイトは情報の完全性を保証しません。",
        ],
      },
      {
        heading: "4. 責任の範囲",
        paragraphs: [
          "当サイトの利用または利用不能により生じた損害について、運営者は責任を負いかねます。閲覧および利用は利用者ご自身の責任でお願いします。",
        ],
      },
      {
        heading: "5. 未成年者の利用について",
        paragraphs: [
          "法令に反する利用、公営競技に関する制限に反する利用、未成年者による不適切な利用は認められません。各自が適用法令を確認のうえ利用してください。",
        ],
      },
    ],
  },
  [`${APP_BASE_PATH}/contact`]: {
    title: "お問い合わせ",
    lead:
      "掲載内容、表記、運営方針、広告掲載、その他ご連絡がある場合は、下記メールアドレスまでお願いします。",
    sections: [
      {
        heading: "連絡先",
        paragraphs: [
          "メール: salvasshaggyya226@gmail.com",
          "内容確認後、順次返信いたします。内容により回答までお時間をいただく場合があります。",
        ],
      },
      {
        heading: "ご連絡時のお願い",
        paragraphs: [
          "対象ページのURL、対象レースID、確認したい内容をできるだけ具体的にご記載ください。",
          "個別の投票判断や利益保証に関するご相談には回答できません。",
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
        <h1>本日の公開レースを読み込んでいます</h1>
        <p>最新の公開データを取得しています。</p>
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
            title="本日のAI分析"
            subtitle="複数の視点を重ねた競馬分析を、レースごとに比較できる形で公開しています。"
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
