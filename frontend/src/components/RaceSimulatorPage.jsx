import React, { useEffect, useMemo, useRef, useState } from "react";
import demo from "../fixtures/raceSimulatorDemo.json";
import calibratedModel from "../fixtures/raceSimulatorCalibrated.json";
import { FRAME_COLORS, formatTime, PACES, sampleSimulation, simulateRace } from "../lib/raceSimulator";
import RacePackView from "./RacePackView";
import RaceCourseView from "./RaceCourseView";
import CourseElevation from "./CourseElevation";
import { COURSE_PROFILES, COURSE_CHECKED_AT, resolveRaceCourse } from "../lib/raceCourses";
import "./RaceSimulatorPage.css";

const INITIAL_SEED = 20261101;
const SPEEDS = [1, 2, 4, 8];

function Icon({ name, size = 18 }) {
  const paths = {
    play: <path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none" />,
    pause: <><path d="M8 5v14M16 5v14" strokeWidth="4" /></>,
    replay: <><path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" /></>,
    shuffle: <><path d="m3 5 4 0 10 14h4m-4-4 4 4-4 3M3 19h4L17 5h4m-4-3 4 3-4 4" /></>,
    arrow: <path d="m14 6-6 6 6 6M8 12h13" />,
    flag: <><path d="M5 21V3m0 1c5-5 9 5 14 0v10c-5 5-9-5-14 0" /></>,
    chart: <><path d="M4 3v17h17M8 15v-3m5 3V8m5 7V5" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v1" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function NumberBadge({ horse, small = false }) {
  return <span className={`sim-number${small ? " sim-number--small" : ""}`} style={{ "--silk": FRAME_COLORS[horse.frame - 1], "--ink": horse.frame === 1 || horse.frame === 5 ? "#24272c" : "#fff" }}>{horse.horse_no}</span>;
}

function Capability({ name, value }) {
  return <div className="sim-capability"><span>{name}</span><div><i style={{ width: `${Math.round(value * 100)}%` }} /></div><b>{Math.round(value * 100)}</b></div>;
}

export default function RaceSimulatorPage() {
  const [raceIndex, setRaceIndex] = useState(0);
  const [pace, setPace] = useState("average");
  const [seed, setSeed] = useState(INITIAL_SEED);
  const [variation, setVariation] = useState(1);
  const [speed, setSpeed] = useState(4);
  const [view, setView] = useState("course");
  const [courseProfileId, setCourseProfileId] = useState(null);
  const [trialDistance, setTrialDistance] = useState(null);
  const [terrainEnabled, setTerrainEnabled] = useState(true);
  const [useCalibrated, setUseCalibrated] = useState(false);
  const [time, setTime] = useState(0);
  const [running, setRunning] = useState(false);
  const [selected, setSelected] = useState(6);
  const [batch, setBatch] = useState(null);
  const [batchProgress, setBatchProgress] = useState(0);
  const batchToken = useRef(0);
  const race = demo.races[raceIndex];
  const courseResolution = useMemo(() => resolveRaceCourse(race.race_info, courseProfileId), [race, courseProfileId]);
  const trialCourse = courseResolution.course;
  const simulationRace = useMemo(() => ({ ...race, race_info: { ...race.race_info, course: trialCourse?.course || race.race_info.course, surface: trialCourse?.surface || race.race_info.surface, source: trialCourse?.group || race.race_info.source, scope_key: trialCourse?.group || race.race_info.scope_key, distance: trialDistance || race.race_info.distance, course_layout: trialCourse?.layout } }), [race, trialCourse, trialDistance]);
  const simulation = useMemo(() => simulateRace(simulationRace, { pace, seed, variation, courseProfileId: trialCourse?.id, terrainEnabled, model: useCalibrated ? calibratedModel : null }), [simulationRace, pace, seed, variation, trialCourse, terrainEnabled, useCalibrated]);
  const frame = useMemo(() => sampleSimulation(simulation, time), [simulation, time]);
  const selectedProfile = simulation.profiles.find((h) => h.horse_no === selected) || simulation.profiles[0];
  const selectedState = frame.horses.find((h) => h.horse_no === selectedProfile.horse_no);
  const leader = frame.order[0];
  const remaining = Math.max(0, Math.ceil((simulation.distance - leader.distance) / 10) * 10);
  const progress = leader.distance / simulation.distance;
  const phase = time === 0 ? "ゲートイン" : leader.finishTime !== null ? "ゴール" : progress < 0.12 ? "スタート" : progress < 0.4 ? "先行争い" : progress < 0.67 ? "中盤の攻防" : "ラストスパート";
  const profiles = useMemo(() => new Map(simulation.profiles.map((h) => [h.horse_no, h])), [simulation]);
  const observed = simulation.events.filter((event) => event.time <= time);
  const withHistory = simulation.profiles.filter((h) => h.runCount > 0).length;
  const withClosing = simulation.profiles.filter((h) => h.closingCount > 0).length;
  const reset = () => { setTime(0); setRunning(false); };

  useEffect(() => {
    document.title = "展開シミュレーター | いかいも競馬";
    return () => { batchToken.current += 1; };
  }, []);
  useEffect(() => {
    reset();
    batchToken.current += 1;
    setBatch(null);
    setBatchProgress(0);
  }, [raceIndex, pace, variation, courseProfileId, trialDistance, terrainEnabled, useCalibrated]);
  useEffect(() => { setCourseProfileId(null); setTrialDistance(null); }, [raceIndex]);
  useEffect(() => { reset(); }, [seed]);
  useEffect(() => {
    if (!running) return undefined;
    let previous;
    let id;
    const animate = (now) => {
      if (previous !== undefined) {
        // Limit jumps after backgrounding the tab. Model time is independent of frame rate.
        const delta = Math.min((now - previous) / 1000, 0.1) * speed;
        setTime((current) => Math.min(simulation.duration, current + delta));
      }
      previous = now;
      id = requestAnimationFrame(animate);
    };
    id = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(id);
  }, [running, speed, simulation]);
  useEffect(() => { if (time >= simulation.duration) setRunning(false); }, [time, simulation.duration]);

  async function runBatch() {
    const token = ++batchToken.current;
    const tallies = Object.fromEntries(simulation.profiles.map((h) => [h.horse_no, { wins: 0, top3: 0 }]));
    setBatch(null);
    setBatchProgress(1);
    for (let start = 0; start < 200; start += 10) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (batchToken.current !== token) return;
      for (let j = start; j < start + 10; j += 1) {
        const result = simulateRace(simulationRace, { seed: INITIAL_SEED + j * 7919, pace, variation, courseProfileId: trialCourse?.id, terrainEnabled, model: useCalibrated ? calibratedModel : null });
        tallies[result.finishOrder[0].horse_no].wins += 1;
        result.finishOrder.slice(0, 3).forEach((h) => { tallies[h.horse_no].top3 += 1; });
      }
      setBatchProgress(start + 10);
    }
    if (batchToken.current === token) { setBatch(tallies); setBatchProgress(0); }
  }

  const info = race.race_info;
  return <main className="sim-page">
    <div className="sim-shell">
      <header className="sim-header">
        <a className="sim-brand" href="/keiba/"><span className="sim-brand-mark"><Icon name="flag" size={22} /></span><span>いかいも競馬<span className="sim-brand-divider" />展開ラボ</span></a>
        <span className="sim-prototype"><i />試作版</span>
        <a href="/keiba/" className="sim-back"><Icon name="arrow" size={16} />レース一覧</a>
      </header>

      <section className="sim-intro">
        <div><p className="sim-eyebrow">もうひとつの展開を、見てみよう。</p><h1>レースを、<span>走らせる。</span></h1><p className="sim-intro-copy">脚質と近走データから描く、一度きりのレース展開。</p></div>
        <div className="sim-intro-note"><Icon name="info" /><p>実際のレース映像・確定予測ではありません。<br />条件と揺らぎによって、結果は変わります。</p></div>
      </section>

      <section className="sim-racebar" aria-label="レース選択">
        <div className="sim-race-label"><span className="sim-section-label">試走するレース</span><label className="sim-select-label"><span className="sim-sr">レースを選択</span><select aria-label="レースを選択" value={raceIndex} onChange={(e) => { setRaceIndex(Number(e.target.value)); setSelected(demo.races[Number(e.target.value)].horses[0].horse_no); }}>
          {demo.races.map((r, i) => <option key={r.race_info.race_id} value={i}>{r.race_info.course} {Number(r.race_info.race_id.slice(-2))}R　{r.race_info.surface} {r.race_info.distance.toLocaleString()}m</option>)}
        </select></label></div>
        <div className="sim-race-meta"><strong>{info.race_name}</strong><span>{info.race_date.replaceAll("-", ".")}<i />{info.track_condition === "稍" ? "稍重" : info.track_condition}<i />{info.weather}<i />{race.horses.length}頭</span></div>
        <div className="sim-input-badge"><i />保存済みの実データを使用</div>
      </section>

      <section className="sim-course-controls" aria-label="走路条件">
        <div className="sim-course-control-heading"><h2>走路条件</h2><span>中央10場・地方14場の平地走路</span><button type="button" onClick={() => { setCourseProfileId(null); setTrialDistance(null); }}>元レースの条件へ</button></div>
        <div className="sim-course-fields">
          <label><span>試走モデル</span><select aria-label="試走モデル" value={useCalibrated ? "calibrated" : "original"} onChange={e=>setUseCalibrated(e.target.value === "calibrated")}><option value="original">従来モデル</option><option value="calibrated">履歴校正モデル（検証中）</option></select></label>
          <label><span>試走コース</span><select aria-label="試走コース" value={trialCourse?.id || ""} onChange={(e) => { const next=COURSE_PROFILES.find(p=>p.id===e.target.value); setCourseProfileId(next.id); const current=simulation.distance; setTrialDistance(next.distances_m.includes(current)?current:next.distances_m.reduce((a,b)=>Math.abs(a-current)<=Math.abs(b-current)?a:b)); }}>
            {!trialCourse && <option value="">走路未設定</option>}
            {[['central','中央'],['local','地方']].map(([group,label])=><optgroup key={group} label={label}>{COURSE_PROFILES.filter(p=>p.group===group).map(p=><option key={p.id} value={p.id}>{p.label}</option>)}</optgroup>)}
          </select></label>
          <label><span>試走距離</span><select aria-label="試走距離" value={simulation.distance} onChange={e=>setTrialDistance(Number(e.target.value))}>
            {[...new Set([...(trialCourse?.distances_m || []),simulation.distance])].sort((a,b)=>a-b).map(d=><option key={d} value={d}>{d.toLocaleString()}m</option>)}
          </select></label>
          <label className="sim-slope-toggle"><input type="checkbox" checked={terrainEnabled && Boolean(trialCourse?.elevation_points)} disabled={!trialCourse?.elevation_points} onChange={e=>setTerrainEnabled(e.target.checked)} /><span>坂の速度・消耗効果を適用</span></label>
        </div>
        {useCalibrated && <p className="sim-course-disclosure">近走の走破時計・末脚・持続力と揺らぎを校正した試験モデルです。体力は実測値ではありません。新しいレースで検証中です。</p>}
        {trialCourse && <div className="sim-course-facts"><span>{trialCourse.direction==='left'?'左回り':trialCourse.direction==='right'?'右回り':'直線'}</span><span>{trialCourse.lap_m?`1周 ${trialCourse.lap_m.toLocaleString()}m`:'直線コース'}</span><span>最終直線 {trialCourse.straight_m.toLocaleString()}m</span><span>高低差 {trialCourse.elevation_m===null?'断面図参照':`${trialCourse.elevation_m}m`}</span><a href={trialCourse.source_url} target="_blank" rel="noreferrer">公式資料 ↗</a></div>}
        {courseResolution.ambiguous && <p className="sim-course-disclosure">保存データに内外回りの指定がないため、{trialCourse?.layout==='outer'?'外回り':'候補走路'}を仮選択しています。上の走路条件で変更できます。</p>}
        {(courseProfileId || trialDistance) && <p className="sim-course-disclosure">コース比較中。出走馬と近走資料は元の{info.course}のレースを使用し、走路・距離だけを変更しています。</p>}
        <p className="sim-course-source-note">寸法は公式公表値（芝は基準コース）。形状・坂は公式図の近似、発走位置は周長から推定。引込線・当日の柵位置は省略しています。確認日 {COURSE_CHECKED_AT.replaceAll('-','.')}</p>
      </section>

      <div className="sim-workspace">
        <section className="sim-stage-panel" aria-label="レース試走">
          <div className="sim-stage-heading"><div className="sim-live"><i className={running ? "is-running" : ""} />{running ? "シミュレーション中" : time >= simulation.duration ? "試走完了" : "展開プレビュー"}</div><span className="sim-stage-time">経過 <b>{formatTime(time)}</b></span></div>
          <div className="sim-track-area"><div className="sim-remaining"><span>先頭の残り距離</span><strong>{remaining.toLocaleString()}<small>m</small></strong></div><div className="sim-view-switch" role="group" aria-label="表示モード"><button type="button" aria-pressed={view === "pack"} className={view === "pack" ? "is-active" : ""} onClick={() => setView("pack")}>馬群拡大</button><button type="button" aria-pressed={view === "course"} className={view === "course" ? "is-active" : ""} onClick={() => setView("course")}>全体コース</button></div><span className="sim-track-caption">{view === "pack" ? "馬間・進路は見やすさのため拡大" : "回り・周長・高低差を走路別に反映"}</span>
            {view === "pack" ? <RacePackView frame={frame} simulation={simulation} selected={selectedProfile.horse_no} onSelect={setSelected} phase={phase} /> : <RaceCourseView frame={frame} simulation={simulation} selected={selectedProfile.horse_no} onSelect={setSelected} phase={phase} />}
          </div>
          <CourseElevation simulation={simulation} frame={frame} selected={selectedProfile.horse_no} />
          <div className="sim-current-event" aria-live="off"><span className="sim-event-dot" /><p>{observed.at(-1)?.text || "準備ができました。再生すると、各馬がそれぞれの展開で走り出します。"}</p></div>
          <div className="sim-transport">
            <div className="sim-timeline-labels"><span>発走</span><span>位置取り</span><span>道中</span><span>スパート</span><span>ゴール</span></div>
            <input className="sim-seek" type="range" min="0" max={simulation.duration} step="0.1" value={time} aria-label="レースの再生位置" aria-valuetext={`${formatTime(time)}、${phase}`} style={{ "--progress": `${time / simulation.duration * 100}%` }} onInput={(e) => { setTime(Number(e.target.value)); setRunning(false); }} onChange={(e) => { setTime(Number(e.target.value)); setRunning(false); }} />
            <div className="sim-controls"><div className="sim-play-group"><button type="button" className="sim-play" onClick={() => { if (time >= simulation.duration) setTime(0); setRunning(!running); }}><Icon name={running ? "pause" : "play"} />{running ? "一時停止" : time >= simulation.duration ? "もう一度再生" : time > 0 ? "再開する" : "レースを再生"}</button><button type="button" className="sim-icon-button" aria-label="同じ試走を最初に戻す" onClick={reset}><Icon name="replay" /></button></div>
              <div className="sim-speed-group"><span>再生速度</span>{SPEEDS.map((s) => <button type="button" key={s} className={s === speed ? "is-active" : ""} aria-pressed={s === speed} onClick={() => setSpeed(s)}>{s}倍</button>)}</div>
            </div>
          </div>
        </section>

        <aside className="sim-order-panel" aria-label="模擬レースの順位">
          <div className="sim-panel-heading"><h2>{time >= simulation.duration ? "模擬レース結果" : "走行順"}</h2><span>{time === 0 ? "馬番順" : leader.finishTime !== null ? "入線中" : "随時更新"}</span></div>
          <div className="sim-order-table"><div className="sim-order-columns"><span>順</span><span>馬名 / 脚質</span><span>{leader.finishTime !== null ? "模擬時計" : "先頭との差"}</span></div>
            {frame.order.map((state, index) => { const horse = profiles.get(state.horse_no); return <button type="button" key={horse.horse_no} className={`sim-order-row${selectedProfile.horse_no === horse.horse_no ? " is-selected" : ""}`} onClick={() => setSelected(horse.horse_no)} aria-pressed={selectedProfile.horse_no === horse.horse_no}>
              <span className={`sim-rank${index < 3 && time > 0 ? " sim-rank--top" : ""}`}>{time === 0 ? "–" : index + 1}</span><NumberBadge horse={horse} small /><span className="sim-horse-copy"><strong>{horse.horse_name}</strong><span>{horse.style}{!horse.runCount && <em>資料少</em>}</span></span><span className="sim-gap">{state.finishTime !== null ? formatTime(state.finishTime) : time === 0 ? "–" : index === 0 ? "先頭" : `${(leader.distance - state.distance).toFixed(1)}m`}</span>
            </button>; })}
          </div><div className="sim-order-foot">番号を選ぶと、馬ごとの推定値を表示</div>
        </aside>
      </div>

      <div className="sim-detail-grid">
        <section className="sim-settings-panel"><div className="sim-panel-heading"><h2>展開を変えてみる</h2><span>条件変更で発走前に戻ります</span></div>
          <label className="sim-field-label">前半のペース</label><div className="sim-pace-buttons">{Object.entries(PACES).map(([key, value]) => <button type="button" key={key} onClick={() => setPace(key)} className={pace === key ? "is-active" : ""} aria-pressed={pace === key}><span className={`sim-pace-mark sim-pace-mark--${key}`}><i /><i /><i /></span>{value.label}</button>)}</div>
          <div className="sim-setting-row"><div><label htmlFor="sim-variation">発走・当日の揺らぎ</label><span>資料が少ない馬ほど振れ幅を大きく</span></div><select id="sim-variation" value={variation} onChange={(e) => setVariation(Number(e.target.value))}><option value="0.4">小さめ</option><option value="1">標準</option><option value="1.7">大きめ</option></select></div>
          <button type="button" className="sim-rerun" onClick={() => setSeed((s) => s + 1)}><Icon name="shuffle" />別の展開で試走<span>↗</span></button><p className="sim-setting-note">再生・巻き戻しは同じ展開。「別の展開」で揺らぎを引き直します。</p>
        </section>

        <section className="sim-horse-panel"><div className="sim-panel-heading"><h2>この馬の走り方</h2><span>推定プロフィール</span></div>
          <div className="sim-selected-horse"><NumberBadge horse={selectedProfile} /><div><h3>{selectedProfile.horse_name}</h3><p>{selectedProfile.style}<i />{selectedProfile.jockey || "騎手情報なし"}<i />近走{selectedProfile.runCount}走</p></div><span className="sim-profile-quality">{selectedProfile.assumptions.length ? "一部仮定" : "近走から推定"}</span></div>
          <div className="sim-capability-grid"><Capability name="基礎能力" value={selectedProfile.ability} /><Capability name="前へ行く力" value={selectedProfile.early} /><Capability name="末脚" value={selectedProfile.closing} /><Capability name="持続力" value={selectedProfile.stamina} /></div>
          <div className="sim-energy"><span>模擬スタミナ残量</span><div><i style={{ width: `${selectedState.energy * 100}%` }} /></div><b>{Math.round(selectedState.energy * 100)}%</b></div>
          <p className="sim-profile-note">{selectedProfile.assumptions.length ? selectedProfile.assumptions.join("。") + "。" : "近走の着順・通過順・同条件の上がりから相対的に推定。"} 数値は実測の身体能力ではありません。</p>
        </section>
      </div>

      <section className="sim-batch-panel"><div className="sim-batch-intro"><div className="sim-section-label">一度の試走、その先へ</div><h2>200回走ったら、どうなる？</h2><p>同じペースと揺らぎ設定で反復。表示値は、この模擬モデル内での出現頻度です。</p></div><button type="button" className="sim-batch-button" onClick={runBatch} disabled={batchProgress > 0}><Icon name="chart" />{batchProgress > 0 ? `試走中 ${batchProgress} / 200` : batch ? "もう一度集計" : "200回の展開を試す"}</button>
        {batch && <div className="sim-batch-results">{Object.entries(batch).sort((a, b) => b[1].top3 - a[1].top3).slice(0, 5).map(([no, values]) => { const horse = profiles.get(Number(no)); return <button type="button" key={no} className="sim-batch-horse" onClick={() => setSelected(Number(no))}><div><NumberBadge horse={horse} small /><strong>{horse.horse_name}</strong><b>{(values.top3 / 2).toFixed(1)}<small>%</small></b></div><div className="sim-batch-bar"><i style={{ width: `${values.top3 / 2}%` }} /></div><span>3着以内 {values.top3}回 · 1着 {values.wins}回</span></button>; })}<p className="sim-batch-caveat">3着以内の出現頻度順。実際の的中率・勝率を示すものではありません。</p></div>}
      </section>

      <details className="sim-method"><summary><Icon name="info" />使用データと、この試作版の仮定<span>＋</span></summary><div className="sim-method-body"><div><h3>使用した実データ</h3><p>{useCalibrated ? "保存済みの出馬表と、対象レースより前の走破時計・上がり・通過順を使用し、脚質は近走から推定しています。" : "保存済みの出馬表と、対象レースより前の日付が確認できる近走、Agent LLMの脚質分析を使用しています。"}</p><ul><li>近走あり：{withHistory} / {race.horses.length}頭</li><li>同馬場・近い距離の上がりあり：{withClosing} / {race.horses.length}頭</li><li>近走記録：計{simulation.profiles.reduce((n, h) => n + h.runCount, 0)}走</li></ul></div><div><h3>仮定していること</h3><p>{useCalibrated ? "速度・消耗・末脚と揺らぎの一部を履歴で校正していますが、実際の位置・体力の計測値ではありません。発走遅れや進路などの仮定は残ります。" : "速度・消耗・加速の係数は展示用の仮定で、実際の位置・体力の計測値ではありません。異なる競走条件の補正は簡略化しています。"}走路と高低断面は公式図を基に近似しています。発走位置・引込線・柵位置は簡略化し、坂による速度・消耗の反応係数は展示用の仮定です。進路取りは簡略化し、馬同士の接触・進路封鎖は再現していません。</p><p>{useCalibrated ? "対象レースの着順、オッズ、人気は入力していません。新しいレースでの検証は少数で、予測精度は確定していません。" : "実際の着順、オッズ、人気は入力していません。モデルの予測精度は未検証です。"}</p></div></div></details>
      <footer className="sim-footer"><span>いかいも競馬 <i />展開ラボ</span><span>模擬展開 #{seed} <i />試作モデル {useCalibrated ? "03" : "02"}</span></footer>
    </div>
  </main>;
}
