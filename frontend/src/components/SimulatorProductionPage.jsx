import React, { useEffect, useState } from "react";
import { appLink, contextFromUrl, fetchSimulatorConfig, validContext } from "../lib/simulatorClient";
import { trackEvent } from "../lib/analytics";
import RaceSimulatorPreview from "./RaceSimulatorPreview";
import AppHeader from "./AppHeader";
import PublicSideNav from "./PublicSideNav";
import SiteFooter from "./SiteFooter";
import { ArrowLeft } from "@phosphor-icons/react";
import "./RaceSimulationExperience.css";

export default function SimulatorProductionPage({ embedded = false, landing = false }) {
  const context = contextFromUrl(), [config, setConfig] = useState(null);
  useEffect(() => { const controller = new AbortController(); fetchSimulatorConfig(controller.signal).then(setConfig).catch(() => {}); return () => controller.abort(); }, []);
  useEffect(() => { document.title = "シミュレーション | いかいも競馬"; }, []);
  if (embedded) return <RaceSimulatorPreview context={context} embedded />;
  const date = validContext(context) ? new Date(`${context.date}T00:00:00+09:00`) : null;
  const boardHref = `/keiba/${date ? `?date=${context.date}` : ""}`;
  return <main className="racing-intel-page sim-production-page">
    <AppHeader showFilters={false} />
    <div className="racing-intel-page__shell racing-intel-page__shell--with-sidebar">
      <PublicSideNav pathname={window.location.pathname} mode="simulator" search={date ? `date=${context.date}` : ""} onApplyFilters={query => { window.location.href = `/keiba/?${query}`; }} />
      <div className="racing-intel-page__main">
        <a className="sim-back-link" href={boardHref}><ArrowLeft size={18} aria-hidden="true" />レース一覧に戻る</a>
        <div className="sim-production-title"><h1>シミュレーション</h1>{date && <time dateTime={context.date}>{date.toLocaleDateString("ja-JP", { timeZone: "Asia/Tokyo", year: "numeric", month: "long", day: "numeric", weekday: "short" })}</time>}</div>
    {landing && <><p>アプリではペースを変えたり、馬ごとの状態を見たりしながら、同じレースを何度でも試走できます。</p><a className="sim-app-link" href={appLink(context)}>アプリでこのレースを開く</a><div className="sim-store-links">{[["android", "Google Play", config?.android_store_url], ["ios", "App Store", config?.ios_store_url]].map(([platform, label, href]) => href && <a key={platform} href={href} onClick={() => trackEvent("sim_store_click", { race_id: context.raceId, input_version: context.inputVersion, model_version: context.modelVersion, source: context.source, platform, protocol_version: 1 })}>{label}</a>)}</div>{config && !config.ios_store_url && <p>iOS版のダウンロード案内は準備中です。</p>}<p>インストール後はこのページに戻り、「アプリでこのレースを開く」を押してください。</p></>}
    {validContext(context) ? <RaceSimulatorPreview context={context} /> : <p>レース詳細から、試走したいレースを選んでください。</p>}
      </div>
    </div>
    <SiteFooter />
  </main>;
}
