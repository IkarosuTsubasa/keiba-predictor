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

function TopFiveIndexList({ items = [] }) {
  if (!items.length) {
    return <p className="morning-preview__top5-empty">AI予測指数を準備中です。</p>;
  }
  return (
    <div className="morning-preview__index-list">
      <div className="morning-preview__index-head">
        <span>馬番</span>
        <span>馬名</span>
        <span>AI指数</span>
      </div>
      {items.slice(0, 6).map((item, index) => (
        <article key={`${item.horse_no}-${index}`} className="morning-preview__index-item">
          <span>{item.horse_no || "-"}</span>
          <strong>{item.horse_name || "-"}</strong>
          <em>{item.support_score || 0}</em>
        </article>
      ))}
    </div>
  );
}

const APP_DOWNLOAD_HREF = "https://x.gd/BDVgd";
const APP_BADGE_SRC = "/keiba/GetItOnGooglePlay_Badge_Web_color_Japanese.png";

export default function MorningPreviewSection({ data, search = "" }) {
  const preview = data?.morning_preview || {};
  const featuredRace = preview?.featured_race || null;
  const ranking = Array.isArray(preview?.confidence_ranking)
    ? preview.confidence_ranking.slice(0, 6)
    : [];

  if (!preview?.available || !featuredRace || !ranking.length) {
    return null;
  }

  return (
    <section className="morning-preview" id="home-morning-preview">
      <div className="morning-preview__head">
        <div>
          <span className="home-section-eyebrow">AI注目</span>
          <h2>今日のAI注目レース</h2>
          <p>本命、自信度、上位5頭、AI予測指数をまとめて確認できます。</p>
        </div>
      </div>

      <div className="morning-preview__hero">
        <article className="morning-preview__featured">
          <div className="morning-preview__featured-top">
            <span>{featuredRace.race_title || "-"}</span>
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
          </div>
          <div className="morning-preview__featured-block">
            <div className="morning-preview__featured-labels">
              <span>上位5頭</span>
            </div>
            <TopFiveIndexList items={featuredRace.top5} />
          </div>
          <a className="morning-preview__link" href={buildRaceHref(featuredRace, search)}>
            詳細を見る
          </a>
        </article>

        <div className="morning-preview__side">
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
        </div>
      </div>

      <article className="morning-preview__app-card morning-preview__app-card--wide">
        <div className="morning-preview__app-row">
          <div className="morning-preview__app-copy">
            <span className="morning-preview__app-eyebrow">アプリ</span>
            <h3>レース直前の最終印を最速でチェック</h3>
            <p>注目レース情報から直前の最終更新まで、完全無料でまとめて確認できます。</p>
          </div>
          <a className="morning-preview__app-link" href={APP_DOWNLOAD_HREF}>
            <img
              className="morning-preview__app-badge"
              src={APP_BADGE_SRC}
              alt="Google Play で手に入れよう"
            />
          </a>
        </div>
      </article>
      
      <div className="morning-preview__footnote">
        <p>注目レースの確認からレース直前の最終チェックまで、アプリですぐ追えます。</p>
      </div>
    </section>
  );
}
