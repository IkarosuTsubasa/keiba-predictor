import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
import { fetchSimulation, validContext, STATUS_TEXT } from "../lib/simulatorClient";
import { trackEvent } from "../lib/analytics";
import "./RaceSimulationExperience.css";
const Experience = lazy(() => import("./RaceSimulationExperience"));

class SimulationBoundary extends React.Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <div className="sim-load-state" role="status"><strong>展開シミュレーション</strong><p>シミュレーションを表示できませんでした。</p><button className="sim-secondary" onClick={() => this.setState({ failed: false })}>再読み込み</button></div>;
    return this.props.children;
  }
}

export default function RaceSimulatorPreview(props) {
  const { context } = props;
  return <SimulationBoundary key={`${context.raceId}:${context.date}:${context.inputVersion || ""}`}><PreviewContent {...props} /></SimulationBoundary>;
}

function PreviewContent({ context, embedded = false }) {
  const element = useRef(null), [visible, setVisible] = useState(embedded), [attempt, setAttempt] = useState(0);
  const [payload, setPayload] = useState(null), [error, setError] = useState(null), [update, setUpdate] = useState(null);
  const key = `${context.raceId}:${context.date}:${context.inputVersion || ""}`;
  const impression = useRef(null);
  const latestController = useRef(null);
  useEffect(() => () => latestController.current?.abort(), [key]);
  const checkLatest = () => {
    latestController.current?.abort();
    const controller = new AbortController(); latestController.current = controller;
    fetchSimulation(context, controller.signal, { latest: true }).then(next => { if (next.input_version !== payload?.input_version) setUpdate(next); }).catch(e => { if (e.name !== "AbortError") setError(e); });
  };
  useEffect(() => {
    if (embedded || !validContext(context) || !element.current) return;
    if (typeof IntersectionObserver === "undefined") { setVisible(true); return; }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) { setVisible(true); observer.disconnect(); }
    }, { rootMargin: "100px" });
    observer.observe(element.current);
    return () => observer.disconnect();
  }, [embedded, key]);
  useEffect(() => {
    if (!visible || !validContext(context)) return;
    const controller = new AbortController();
    setPayload(null); setError(null); setUpdate(null);
    fetchSimulation(context, controller.signal).then(data => { setPayload(data); if (!embedded && impression.current !== key && !new URLSearchParams(location.search).has("app")) { impression.current = key; trackEvent("sim_preview_impression", { race_id: data.race_id, input_version: data.input_version, model_version: data.active_model_version, source: context.source, platform: "web", protocol_version: 1 }); } }).catch(e => { if (e.name !== "AbortError") setError(e); });
    return () => controller.abort();
  }, [visible, key, attempt]);
  useEffect(() => {
    if (!payload) return;
    const controller = new AbortController();
    const check = () => { if (!document.hidden) fetchSimulation(context, controller.signal, { latest: true }).then(next => { if (next.input_version !== payload.input_version) { setUpdate(next); if (embedded && window.SimulationChannel) window.SimulationChannel.postMessage(JSON.stringify({ version: 1, type: "inputUpdate", raceId: payload.race_id, date: payload.race_date, inputVersion: payload.input_version, modelVersion: payload.active_model_version, data: { inputVersion: next.input_version } })); } }).catch(() => {}); };
    const id = setInterval(check, 60000);
    return () => { clearInterval(id); controller.abort(); };
  }, [payload, key]);
  if (!validContext(context)) return <section aria-label="展開シミュレーション" className="sim-load-state"><strong>展開シミュレーション</strong><p>{STATUS_TEXT.preparing}</p></section>;
  const available = payload && ["ready", "partial"].includes(payload.availability);
  return <section ref={element} aria-label="展開シミュレーション">
    {update && <div className="sim-update-notice">新しいデータがあります。<button onClick={() => { if (embedded && window.SimulationChannel) window.SimulationChannel.postMessage(JSON.stringify({ version: 1, type: "inputUpdate", raceId: payload.race_id, date: payload.race_date, inputVersion: payload.input_version, modelVersion: payload.active_model_version, data: { inputVersion: update.input_version } })); else { setPayload(update); setUpdate(null); } }}>更新して試走する</button></div>}
    {available ? <Suspense fallback={<div className="sim-load-state">シミュレーションを準備しています。</div>}><Experience key={payload.input_version} payload={payload} context={context} embedded={embedded} onUpdate={checkLatest} /></Suspense> : <div className="sim-load-state"><strong>展開シミュレーション</strong><p>{error?.message || (payload ? STATUS_TEXT[payload.availability] : "データを準備しています。")}</p>{visible && <button className="sim-secondary" onClick={() => setAttempt(a => a + 1)}>再読み込み</button>}{embedded && error?.code === "version_not_found" && <a href={`/keiba/simulator/embed?race_id=${context.raceId}&date=${context.date}`}>最新のデータを開く</a>}</div>}
  </section>;
}
