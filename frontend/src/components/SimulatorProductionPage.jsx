import React, { useEffect, useState } from "react";
import { appLink, contextFromUrl, fetchSimulatorConfig, validContext } from "../lib/simulatorClient";
import { trackEvent } from "../lib/analytics";
import RaceSimulatorPreview from "./RaceSimulatorPreview";
import "./RaceSimulationExperience.css";

export default function SimulatorProductionPage({ embedded = false, landing = false }) {
  const context = contextFromUrl(), [config, setConfig] = useState(null);
  useEffect(() => { const controller = new AbortController(); fetchSimulatorConfig(controller.signal).then(setConfig).catch(() => {}); return () => controller.abort(); }, []);
  useEffect(() => { document.title = "シミュレーション | いかいも競馬"; }, []);
  if (embedded) return <RaceSimulatorPreview context={context} embedded />;
  return <main className="sim-production-page"><a href="/keiba/">レース一覧に戻る</a><h1>シミュレーション</h1>
    {landing && <><p>アプリではペースを変えたり、馬ごとの状態を見たりしながら、同じレースを何度でも試走できます。</p><a className="sim-app-link" href={appLink(context)}>アプリでこのレースを開く</a><div className="sim-store-links">{[["android", "Google Play", config?.android_store_url], ["ios", "App Store", config?.ios_store_url]].map(([platform, label, href]) => href && <a key={platform} href={href} onClick={() => trackEvent("sim_store_click", { race_id: context.raceId, input_version: context.inputVersion, model_version: context.modelVersion, source: context.source, platform, protocol_version: 1 })}>{label}</a>)}</div>{config && !config.ios_store_url && <p>iOS版のダウンロード案内は準備中です。</p>}<p>インストール後はこのページに戻り、「アプリでこのレースを開く」を押してください。</p></>}
    {validContext(context) ? <RaceSimulatorPreview context={context} /> : <p>レース詳細から、試走したいレースを選んでください。</p>}
  </main>;
}
