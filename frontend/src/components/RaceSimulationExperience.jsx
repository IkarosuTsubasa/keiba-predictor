import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { sampleSimulation, simulateRace, formatTime, FRAME_COLORS } from "../lib/raceSimulator";
import { ArrowCounterClockwise, CaretRight, DeviceMobile, Pause, Play } from "@phosphor-icons/react";
import { COURSE_PROFILES } from "../lib/raceCourses";
import { ACTIVE_MODEL, appLink, engineRace, validateSettings } from "../lib/simulatorClient";
import { trackEvent } from "../lib/analytics";
import RaceSimulationRenderer from "./RaceSimulationRenderer";
import "./RaceSimulatorPage.css";
import "./RaceSimulationExperience.css";

export default function RaceSimulationExperience({ payload, context, embedded = false, onUpdate }) {
  const [settings, setSettings] = useState({ seed: context.seed ?? payload.default_seed, pace: "average", variation: 1, speed: 4, view: "course", distance: null, courseProfileId: null });
  const [time, setTime] = useState(0), [running, setRunning] = useState(false);
  const [selected, setSelected] = useState(null), [batch, setBatch] = useState(null), [completed, setCompleted] = useState(0);
  const [active, setActive] = useState(true);
  const [restoration, setRestoration] = useState(0);
  const worker = useRef(null), command = useRef(null), current = useRef(null), lastState = useRef(0), started = useRef(false), finished = useRef(false), pendingRestore = useRef(null);
  const race = useMemo(() => {
    const original = engineRace(payload), course = COURSE_PROFILES.find(c => c.id === settings.courseProfileId);
    return { ...original, race_info: { ...original.race_info, ...(course ? { course: course.course, surface: course.surface, source: course.group, scope_key: course.group, course_layout: course.layout } : {}), distance: settings.distance || original.race_info.distance } };
  }, [payload, settings.courseProfileId, settings.distance]);
  const simulation = useMemo(() => simulateRace(race, settings), [race, settings.seed, settings.pace, settings.variation, settings.courseProfileId]);
  const frame = useMemo(() => sampleSimulation(simulation, time), [simulation, time]);
  const selectedNo = selected ?? race.horses[0].horse_no;
  const identity = { raceId: payload.race_id, date: payload.race_date, inputVersion: payload.input_version, modelVersion: ACTIVE_MODEL };
  const emit = useCallback((type, data = {}, requestId = null) => {
    if (embedded && window.SimulationChannel) window.SimulationChannel.postMessage(JSON.stringify({ version: 1, type, ...identity, requestId, data }));
  }, [embedded, payload]);
  const event = useCallback((name, data = {}) => {
    const dimensions = { race_id: payload.race_id, input_version: payload.input_version, model_version: ACTIVE_MODEL, source: context.source, protocol_version: 1, ...data };
    if (embedded) emit("analytics", { name, ...dimensions });
    else if (!new URLSearchParams(window.location.search).has("app")) trackEvent(name, { ...dimensions, platform: "web" });
  }, [payload, context.source, embedded, emit]);
  const cancelBatch = useCallback(() => { worker.current?.terminate(); worker.current = null; setCompleted(0); }, []);
  const reset = useCallback(() => { setRunning(false); setTime(0); }, []);
  useEffect(() => { reset(); cancelBatch(); setBatch(null); finished.current = false; if (pendingRestore.current) { setTime(Math.min(simulation.duration, Math.max(0, pendingRestore.current.time || 0))); setSelected(pendingRestore.current.selected || null); pendingRestore.current = null; } }, [simulation, reset, cancelBatch, restoration]);
  useEffect(() => () => worker.current?.terminate(), []);
  useEffect(() => {
    const stop = () => { if (document.hidden) { setRunning(false); cancelBatch(); } };
    document.addEventListener("visibilitychange", stop);
    return () => document.removeEventListener("visibilitychange", stop);
  }, [cancelBatch]);
  useEffect(() => {
    if (!running || !active) return;
    let id, previous;
    const animate = now => { if (previous != null) setTime(t => Math.min(simulation.duration, t + Math.min((now - previous) / 1000, .1) * settings.speed)); previous = now; id = requestAnimationFrame(animate); };
    id = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(id);
  }, [running, active, simulation, settings.speed]);
  useEffect(() => {
    if (!running || time < simulation.duration) return;
    setRunning(false);
    if (!finished.current) {
      finished.current = true;
      event(embedded ? "sim_app_complete" : "sim_preview_complete");
      emit("complete", { finishOrder: simulation.finishOrder });
    }
  }, [running, time, simulation, event, emit, embedded]);
  const play = () => { if (!active) return; if (time >= simulation.duration) setTime(0); setRunning(true); if (!started.current) { started.current = true; event(embedded ? "sim_app_start" : "sim_preview_start"); } };
  const runBatch = requestId => {
    cancelBatch(); setBatch(null); setCompleted(1);
    const taskId = `${payload.input_version}:${Date.now()}`;
    const task = new Worker(new URL("../lib/simulatorBatch.worker.js", import.meta.url), { type: "module" });
    worker.current = task;
    task.onmessage = ({ data }) => {
      if (worker.current !== task || data.taskId !== taskId) return;
      setCompleted(data.completed);
      emit(data.type === "complete" ? "batchComplete" : "batchProgress", data, requestId);
      if (data.type === "complete") { setBatch(data.tallies); setCompleted(0); task.terminate(); worker.current = null; event("sim_batch_complete", { count: 200 }); }
    };
    task.onerror = () => { if (worker.current === task) { cancelBatch(); emit("error", { message: "集計できませんでした。もう一度お試しください。" }, requestId); } };
    task.postMessage({ race, settings, taskId });
  };
  current.current = { time, settings, running, simulation, selected: selectedNo, batch };
  command.current = message => {
    if (message.version !== 1 || message.raceId !== payload.race_id || message.date !== payload.race_date || message.inputVersion !== payload.input_version || message.modelVersion !== ACTIVE_MODEL || typeof message.requestId !== "string") return;
    try {
      const data = message.data || {};
      switch (message.type) {
        case "play": play(); break;
        case "pause": setRunning(false); break;
        case "seek": if (!Number.isFinite(data.time) || data.time < 0 || data.time > simulation.duration) throw new Error(); setTime(data.time); setRunning(false); break;
        case "settings": {
          const next = validateSettings(data);
          if (next.courseProfileId && !COURSE_PROFILES.some(c => c.id === next.courseProfileId)) throw new Error();
          setSettings(s => ({ ...s, ...next })); event("sim_settings_change"); break;
        }
        case "restore": {
          const next = validateSettings(data.settings || {});
          if (next.courseProfileId && !COURSE_PROFILES.some(c => c.id === next.courseProfileId)) throw new Error();
          pendingRestore.current = { time: Number.isFinite(data.time) ? data.time : 0, selected: data.selected };
          // New settings object must rebuild the simulation even if values match.
          setSettings(s => ({ ...s, ...next }));
          setRestoration(v => v + 1);
          setRunning(false); break;
        }
        case "reset": setSettings({ seed: payload.default_seed, pace: "average", variation: 1, speed: 4, view: "course", distance: null, courseProfileId: null }); reset(); break;
        case "select": if (!race.horses.some(h => h.horse_no === data.horseNo)) throw new Error(); setSelected(data.horseNo); break;
        case "batch": if (active) runBatch(message.requestId); break;
        case "cancelBatch": cancelBatch(); break;
        case "active": if (typeof data.active !== "boolean") throw new Error(); setActive(data.active); if (!data.active) { setRunning(false); cancelBatch(); } break;
        case "snapshot": break;
        default: throw new Error();
      }
      emit("ack", { command: message.type }, message.requestId);
    } catch { emit("error", { message: "設定を確認して、もう一度お試しください。" }, message.requestId); }
  };
  useEffect(() => {
    if (!embedded || !window.SimulationChannel) return;
    window.keibaSimulator = { receive: message => command.current?.(message) };
    emit("ready", { duration: simulation.duration, settings: current.current.settings, horses: simulation.profiles, courses: COURSE_PROFILES.map(c => ({ id: c.id, label: c.label, distances: c.distances_m })), availability: payload.availability });
    return () => { delete window.keibaSimulator; };
  }, [embedded, payload, emit]);
  useEffect(() => {
    if (!embedded) return;
    const now = performance.now();
    if (now - lastState.current > 220 || !running) { lastState.current = now; emit("state", { time, running, duration: simulation.duration, settings, selected: selectedNo, horses: frame.horses, order: frame.order, profiles: simulation.profiles }); }
  }, [time, running, settings, selectedNo, embedded, simulation, emit]);
  const selectHorse = no => { setSelected(no); if (embedded) emit("selected", { horseNo: no }); };
  // The engine uses floating-point ticks. An integer range makes End reach its exact duration.
  const progressMax = Math.round(simulation.duration * 10);
  const runningOrder = !embedded && <aside className="sim-preview-order" aria-label="走行順">
    <h3>{time === 0 ? "出走馬" : "走行順"}</h3>
    <ol>
      {frame.order.slice(0, 5).map((state, index) => {
        const horse = simulation.profiles.find(h => h.horse_no === state.horse_no);
        return <li key={horse.horse_no}>
          <button type="button" aria-pressed={selectedNo === horse.horse_no} aria-label={`${horse.horse_no}番 ${horse.horse_name}を選択`} onClick={() => selectHorse(horse.horse_no)}>
            <span className="sim-order-place">{time === 0 ? "–" : index + 1}</span>
            <span className={`sim-horse-number sim-horse-number--${horse.frame}`} style={{ backgroundColor: FRAME_COLORS[horse.frame - 1] }}>{horse.horse_no}</span>
            <span className="sim-order-name">{horse.horse_name}</span>
          </button>
        </li>;
      })}
    </ol>
  </aside>;
  return <div className={`sim-experience${embedded ? " sim-experience--embed" : ""}`}>
    {!embedded && <div className="sim-preview-heading"><h2><span>{payload.race_info.course} {Number(payload.race_id.slice(-2))}R</span>{payload.race_info.race_name && <span>{payload.race_info.race_name}</span>}<span className="sim-race-conditions">{payload.race_info.surface} {payload.race_info.distance.toLocaleString("ja-JP")}m・{race.horses.length}頭</span></h2><span className="sim-version">{payload.availability === "partial" ? "一部のデータを推定" : "データ取得済み"}</span></div>}
    <RaceSimulationRenderer simulation={simulation} frame={frame} time={time} selected={selectedNo} onSelect={selectHorse} view={settings.view} companion={runningOrder} presentation={!embedded} />
    {!embedded && <>
      <div className="sim-preview-transport">
        <button type="button" className="sim-preview-play" onClick={() => running ? setRunning(false) : play()}>{running ? <Pause size={20} weight="fill" aria-hidden="true" /> : <Play size={20} weight="fill" aria-hidden="true" />}{running ? "一時停止" : time >= simulation.duration ? "もう一度再生" : "レースを再生"}</button>
        <button type="button" className="sim-secondary" onClick={reset}><ArrowCounterClockwise size={18} aria-hidden="true" />最初に戻す</button>
        <input
          className="sim-preview-progress"
          type="range"
          min={0}
          max={progressMax}
          step={1}
          value={Math.round(time * 10)}
          aria-label="再生位置"
          aria-valuetext={formatTime(time)}
          onChange={event => {
            const position = event.currentTarget.valueAsNumber;
            setRunning(false);
            setTime(position === progressMax ? simulation.duration : position / 10);
          }}
        />
        <span className="sim-preview-clock">{formatTime(time)} / {formatTime(simulation.duration)}</span>
      </div>
      {time >= simulation.duration && <div className="sim-preview-finish"><strong>模擬レースの入線順</strong><ol>{simulation.finishOrder.slice(0, 3).map(h => <li key={h.horse_no}>{h.horse_no}番 {race.horses.find(p => p.horse_no === h.horse_no)?.horse_name} · {formatTime(h.finishTime)}</li>)}</ol></div>}
      <p className="sim-preview-note">近走データを基にした模擬レースです。位置取りやスタミナは推定値です。実際の結果とは異なる場合があります。</p>
      <a className="sim-app-link" href={appLink({ ...context, ...identity, seed: settings.seed })} onClick={() => event("sim_app_cta_click")}><DeviceMobile size={30} aria-hidden="true" /><div>アプリで展開を変える <span>ペースの変更・馬ごとの状態・200回の試走</span></div><CaretRight size={20} aria-hidden="true" /></a>
      <div className="sim-preview-source"><p className="sim-version">{payload.source_as_of ? "データ取得" : "取得時刻不明・データ公開"} {new Date(payload.source_as_of || payload.published_at).toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" })} · {payload.input_version.slice(0, 8)}</p>
      {onUpdate && <button type="button" className="sim-source-refresh" onClick={onUpdate}>最新のデータを確認</button>}</div>
    </>}
  </div>;
}
