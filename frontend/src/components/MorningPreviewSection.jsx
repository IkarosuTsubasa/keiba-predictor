import React from "react";
import { buildRaceDetailHref } from "../lib/publicRace";

function buildRaceHref(race, search = "") {
  return buildRaceDetailHref(race, search);
}

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function TopFiveLine({ items = [] }) {
  if (!items.length) {
    return <p className="morning-preview__top5-empty">上位候補を準備中です。</p>;
  }
  return (
    <div className="morning-preview__top5">
      {items.slice(0, 5).map((item, index) => (
        <span key={`${item.horse_no}-${index}`}>
          <strong>{item.horse_no || "-"}</strong>
          <em>{item.support_score || 0}</em>
        </span>
      ))}
    </div>
  );
}

const APP_DOWNLOAD_HREF = "https://x.gd/BDVgd";

export default function MorningPreviewSection({ data, search = "" }) {
  const preview = data?.morning_preview || {};
  const featuredRace = preview?.featured_race || null;
  const ranking = Array.isArray(preview?.confidence_ranking) ? preview.confidence_ranking : [];

  if (!preview?.available || !featuredRace || !ranking.length) {
    return null;
  }

  return (
    <section className="morning-preview" id="home-morning-preview">
      <div className="morning-preview__head">
        <div>
          <span className="home-section-eyebrow">朝版速報</span>
          <h2>今日のAI注目レース</h2>
          <p>v1 と 極 KIWAMI の前5候補を束ねて、朝時点の注目度と自信度を先出しします。</p>
        </div>
      </div>

      <div className="morning-preview__hero">
        <article className="morning-preview__featured">
          <div className="morning-preview__featured-top">
            <span>{featuredRace.race_title || "-"}</span>
            <strong>{featuredRace.confidence_label || "-"}</strong>
          </div>
          <div className="morning-preview__featured-main">
            <div>
              <small>AI本命</small>
              <strong>{featuredRace.main_horse_no || "-"}</strong>
            </div>
            <div>
              <small>自信度</small>
              <strong>{formatScore(featuredRace.confidence_score)}</strong>
            </div>
            <div>
              <small>一致度</small>
              <strong>{formatScore(featuredRace.agreement_score)}</strong>
            </div>
          </div>
          <p>{featuredRace.summary_text || "上位候補を比較中"}</p>
          <TopFiveLine items={featuredRace.top5} />
          <a className="morning-preview__link" href={buildRaceHref(featuredRace, search)}>
            詳細を見る
          </a>
        </article>

        <div className="morning-preview__side">
          <article className="morning-preview__app-card">
            <div className="morning-preview__subhead">
              <span>アプリ</span>
              <h3>レース直前のAI最終予想</h3>
            </div>
            <ul className="morning-preview__app-points">
              <li>6つの定量モデルを確認</li>
              <li>レース直前の最終印を比較</li>
              <li>すべて無料でチェック</li>
            </ul>
            <a className="morning-preview__app-link" href={APP_DOWNLOAD_HREF}>
              Androidアプリをダウンロード
            </a>
          </article>
        </div>
      </div>

      <article className="morning-preview__ranking-card">
        <div className="morning-preview__subhead">
          <span>ランキング</span>
          <h3>AI自信度ランキング</h3>
        </div>
        <ol className="morning-preview__ranking">
          {ranking.map((item) => (
            <li key={item.run_id}>
              <a href={buildRaceHref(item, search)}>
                <span>{item.race_title || "-"}</span>
                <strong>{item.main_horse_no || "-"}</strong>
                <em>{formatScore(item.confidence_score)}</em>
              </a>
            </li>
          ))}
        </ol>
      </article>
      
      <div className="morning-preview__footnote">
        <p>朝は注目度を確認して、レース直前の最終判断はアプリで追う流れがおすすめです。</p>
      </div>
    </section>
  );
}
