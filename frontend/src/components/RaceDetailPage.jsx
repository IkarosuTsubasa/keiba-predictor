import React, { useMemo } from "react";
import AutoFitLine from "./AutoFitLine";
import { resolvePublicDecision } from "../lib/confidencePolicy";
import {
  APP_BASE_PATH,
  MARK_ORDER,
  buildRaceDetailHref,
  buildPredictorConsensusSummary,
  formatRaceBadges,
  matchRaceIdentifier,
  parseMarks,
  parseResultEntries,
  pickHorse,
  raceIdentifier,
} from "../lib/publicRace";

const MAIN_MARK = MARK_ORDER[0];
const MARK_WEIGHT = { "◎": 5, "○": 4, "▲": 3, "△": 2, "☆": 1 };
const APP_DOWNLOAD_HREF = "https://x.gd/BDVgd";
const APP_BADGE_SRC = "/keiba/GetItOnGooglePlay_Badge_Web_color_Japanese.png";
const PREDICTOR_LABELS = {
  main: "ゲート",
  v2_opus: "ストライド",
  v3_premium: "伯楽",
  v4_gemini: "馬場眼",
  v5_stacking: "フュージョン",
  v6_kiwami: "極 KIWAMI",
};
const PREDICTOR_ORDER = ["main", "v2_opus", "v3_premium", "v4_gemini", "v5_stacking", "v6_kiwami"];
const COURSE_ACCENTS = [
  { accent: "#15803d", soft: "#eefdf2" },
  { accent: "#2563eb", soft: "#eff6ff" },
  { accent: "#4338ca", soft: "#eef2ff" },
  { accent: "#b45309", soft: "#fff7ed" },
  { accent: "#0f766e", soft: "#ecfeff" },
  { accent: "#be123c", soft: "#fff1f2" },
];
const COURSE_ACCENT_BY_LOCATION = {
  門別: { accent: "#15803d", soft: "#eefdf2" },
  水沢: { accent: "#2563eb", soft: "#eff6ff" },
  大井: { accent: "#4338ca", soft: "#eef2ff" },
  金沢: { accent: "#b45309", soft: "#fff7ed" },
  東京: { accent: "#1d4ed8", soft: "#eff6ff" },
  京都: { accent: "#b45309", soft: "#fff7ed" },
  阪神: { accent: "#7c3aed", soft: "#f5f3ff" },
  中山: { accent: "#0f766e", soft: "#ecfeff" },
  中京: { accent: "#be123c", soft: "#fff1f2" },
  札幌: { accent: "#0369a1", soft: "#f0f9ff" },
  函館: { accent: "#047857", soft: "#ecfdf5" },
  新潟: { accent: "#0d9488", soft: "#f0fdfa" },
  福島: { accent: "#c2410c", soft: "#fff7ed" },
  小倉: { accent: "#a21caf", soft: "#fdf4ff" },
};

function buildBackHref(search) {
  const query = String(search || "").replace(/^\?/, "");
  return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
}

function displayOrderValue(race) {
  const value = Number(race?.display_order);
  return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER;
}

function courseAccent(location, index) {
  return COURSE_ACCENT_BY_LOCATION[location] || COURSE_ACCENTS[index % COURSE_ACCENTS.length];
}

function raceLocation(race) {
  return String(race?.location || "").trim();
}

function raceTitleText(race) {
  return String(race?.display_header?.title || race?.race_title || race?.race_id || "").trim();
}

function extractClockText(value) {
  const matched = String(value || "").trim().match(/(\d{2}:\d{2})/);
  return matched ? matched[1] : "";
}

function raceClockMinutes(race) {
  const clockText = extractClockText(race?.scheduled_off_time);
  if (!clockText) return Number.MAX_SAFE_INTEGER;
  const [hourText, minuteText] = clockText.split(":");
  const hour = Number(hourText);
  const minute = Number(minuteText);
  return Number.isFinite(hour) && Number.isFinite(minute)
    ? hour * 60 + minute
    : Number.MAX_SAFE_INTEGER;
}

function extractRaceNoText(race) {
  const candidates = [
    race?.display_header?.title,
    race?.display_header?.detail_title,
    race?.race_title,
    race?.race_id,
    race?.card_id,
    race?.run_id,
  ];
  for (const value of candidates) {
    const source = String(value || "").trim();
    const matched = source.match(/(\d+)\s*R/i);
    if (matched) {
      return `${Number(matched[1])}R`;
    }
  }
  const raceId = String(race?.race_id || "").trim();
  if (/^\d+$/.test(raceId) && raceId.length >= 2) {
    const raceNo = Number(raceId.slice(-2));
    if (Number.isFinite(raceNo) && raceNo > 0) {
      return `${raceNo}R`;
    }
  }
  return "";
}

function raceNoValue(race) {
  const matched = extractRaceNoText(race).match(/\d+/);
  const value = Number(matched?.[0]);
  return Number.isFinite(value) && value > 0 ? value : Number.MAX_SAFE_INTEGER;
}

function raceNavigationKey(race) {
  const id = raceIdentifier(race);
  if (id) return `id:${id}`;
  const location = raceLocation(race);
  const raceNo = extractRaceNoText(race);
  const clockText = extractClockText(race?.scheduled_off_time);
  if (location && raceNo) {
    return `slot:${location}:${raceNo}:${clockText}`;
  }
  const title = raceTitleText(race);
  return title ? `title:${title}` : "";
}

function isSameRace(left, right) {
  if (!left || !right) return false;
  const leftId = raceIdentifier(left);
  const rightId = raceIdentifier(right);
  if (leftId && matchRaceIdentifier(right, leftId)) return true;
  if (rightId && matchRaceIdentifier(left, rightId)) return true;
  const leftKey = raceNavigationKey(left);
  return Boolean(leftKey && leftKey === raceNavigationKey(right));
}

function compareRacesForSequence(left, right) {
  const leftNo = raceNoValue(left);
  const rightNo = raceNoValue(right);
  if (leftNo !== rightNo) return leftNo - rightNo;
  const leftClock = raceClockMinutes(left);
  const rightClock = raceClockMinutes(right);
  if (leftClock !== rightClock) return leftClock - rightClock;
  const leftOrder = displayOrderValue(left);
  const rightOrder = displayOrderValue(right);
  if (leftOrder !== rightOrder) return leftOrder - rightOrder;
  return raceTitleText(left).localeCompare(raceTitleText(right), "ja");
}

function sortRacesForSequence(races) {
  return [...(races || [])].sort(compareRacesForSequence);
}

function sortRacesForCourseTabs(races) {
  return [...(races || [])].sort((left, right) => {
    const leftOrder = displayOrderValue(left);
    const rightOrder = displayOrderValue(right);
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    return compareRacesForSequence(left, right);
  });
}

function raceNavigationLabel(race) {
  const raceNo = extractRaceNoText(race);
  const title = raceTitleText(race);
  if (raceNo && title && !title.includes(raceNo)) {
    return `${raceNo} ${title}`;
  }
  return title || raceNo || "-";
}

function raceNavigationMeta(race) {
  return [
    extractClockText(race?.scheduled_off_time),
    race?.distance_label,
    race?.display_status?.label,
  ]
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .join(" / ");
}

function buildDetailNavigation(currentRace, races) {
  if (!currentRace) return null;
  const sourceRaces = Array.isArray(races) ? races.filter(Boolean) : [];
  const includesCurrent = sourceRaces.some((item) => isSameRace(item, currentRace));
  const pool = includesCurrent ? sourceRaces : [currentRace, ...sourceRaces].filter(Boolean);
  const selectedRace = sourceRaces.find((item) => isSameRace(item, currentRace)) || currentRace;
  const currentLocation = raceLocation(selectedRace);

  const sameLocationRaces = sortRacesForSequence(
    pool.filter((item) => raceLocation(item) && raceLocation(item) === currentLocation),
  );
  const currentIndex = sameLocationRaces.findIndex((item) => isSameRace(item, selectedRace));
  const previousRace = currentIndex > 0 ? sameLocationRaces[currentIndex - 1] : null;
  const nextRace =
    currentIndex >= 0 && currentIndex < sameLocationRaces.length - 1
      ? sameLocationRaces[currentIndex + 1]
      : null;

  const groups = [];
  const groupIndexByLocation = new Map();
  for (const item of sortRacesForCourseTabs(pool)) {
    const location = raceLocation(item);
    if (!location) continue;
    let groupIndex = groupIndexByLocation.get(location);
    if (groupIndex === undefined) {
      groupIndex = groups.length;
      groupIndexByLocation.set(location, groupIndex);
      groups.push({ location, races: [] });
    }
    groups[groupIndex].races.push(item);
  }

  const currentRaceNo = raceNoValue(selectedRace);
  const courseGroups = groups.map((group, index) => {
    const sortedGroupRaces = sortRacesForSequence(group.races);
    const sameNoRace =
      currentRaceNo !== Number.MAX_SAFE_INTEGER
        ? sortedGroupRaces.find((item) => raceNoValue(item) === currentRaceNo)
        : null;
    const active = group.location === currentLocation;
    return {
      location: group.location,
      accent: courseAccent(group.location, index),
      count: sortedGroupRaces.length,
      active,
      targetRace: active ? selectedRace : sameNoRace || sortedGroupRaces[0] || null,
    };
  });

  return {
    currentRace: selectedRace,
    previousRace,
    nextRace,
    courseGroups,
    hasNavigation: Boolean(previousRace || nextRace || courseGroups.length > 1),
  };
}

function buildFullReleaseText(scheduledOffTime) {
  const offClock = extractClockText(scheduledOffTime);
  if (!offClock) {
    return "完全公開準備中";
  }
  const [hourText, minuteText] = offClock.split(":");
  const totalMinutes = Math.max(
    0,
    Number(hourText || 0) * 60 + Number(minuteText || 0) - 25,
  );
  const publishHour = String(Math.floor(totalMinutes / 60)).padStart(2, "0");
  const publishMinute = String(totalMinutes % 60).padStart(2, "0");
  return `${publishHour}:${publishMinute}頃完全公開`;
}

function buildHorseSignalRows(predictorCards = []) {
  const tally = new Map();
  const modelCount = Array.isArray(predictorCards) ? predictorCards.filter(Boolean).length : 0;

  for (const card of predictorCards || []) {
    const marks = parseMarks(card?.marks_text);
    const topHorses = Array.isArray(card?.top_horses) && card.top_horses.length
      ? card.top_horses.filter(Boolean).slice(0, 5)
      : marks.map((item, index) => ({
        horse_no: String(item?.horseNo || "").trim(),
        horse_name: "",
        top3_prob_model: 0,
        rank_score_norm: Math.max(0, 1 - index * 0.18),
      }));
    const markMap = new Map(
      marks
        .filter(Boolean)
        .map((item) => [String(item?.horseNo || "").trim(), item?.symbol])
        .filter(([horseNo, symbol]) => horseNo && symbol),
    );
    if (!marks.length && !topHorses.length) continue;
    const seen = new Set();

    for (const item of topHorses) {
      const horseNo = String(item?.horse_no || item?.horseNo || "").trim();
      if (!horseNo) continue;
      const symbol = markMap.get(horseNo) || "";
      const entry = tally.get(horseNo) || {
        horseNo,
        horseName: "",
        score: 0,
        supportCount: 0,
        mainCount: 0,
        top3ProbTotal: 0,
        rankScoreTotal: 0,
        entryCount: 0,
      };
      entry.horseName = entry.horseName || String(item?.horse_name || item?.horseName || "").trim();
      entry.score += MARK_WEIGHT[symbol] || 0;
      entry.top3ProbTotal += Math.max(0, Number(item?.top3_prob_model || 0) || 0);
      entry.rankScoreTotal += Math.max(0, Number(item?.rank_score_norm || 0) || 0);
      entry.entryCount += 1;
      if (symbol === MAIN_MARK) {
        entry.mainCount += 1;
      }
      if (!seen.has(horseNo)) {
        entry.supportCount += 1;
        seen.add(horseNo);
      }
      tally.set(horseNo, entry);
    }
  }

  const rows = [...tally.values()].sort(
    (left, right) =>
      right.score - left.score ||
      right.mainCount - left.mainCount ||
      right.supportCount - left.supportCount ||
      Number(left.horseNo) - Number(right.horseNo),
  );

  return rows.map((row) => ({
    ...row,
    horseName: row.horseName || "-",
    avgMarkStrength:
      row.supportCount > 0 ? Math.max(0, Math.min(1, row.score / (row.supportCount * 5))) : 0,
    avgTop3Prob:
      row.entryCount > 0 ? Math.max(0, Math.min(1, row.top3ProbTotal / row.entryCount)) : 0,
    avgRankScore:
      row.entryCount > 0 ? Math.max(0, Math.min(1, row.rankScoreTotal / row.entryCount)) : 0,
    aiIndex: Math.max(
      1,
      Math.min(
        99,
        Math.round(
          42 +
            20 * (row.supportCount > 0 ? Math.max(0, Math.min(1, row.score / (row.supportCount * 5))) : 0) +
            16 * (row.entryCount > 0 ? Math.max(0, Math.min(1, row.top3ProbTotal / row.entryCount)) : 0) +
            12 * (row.entryCount > 0 ? Math.max(0, Math.min(1, row.rankScoreTotal / row.entryCount)) : 0) +
            10 * (modelCount > 0 ? row.supportCount / modelCount : 0),
        ),
      ),
    ),
  })).sort(
    (left, right) =>
      right.aiIndex - left.aiIndex ||
      right.score - left.score ||
      right.mainCount - left.mainCount ||
      Number(left.horseNo) - Number(right.horseNo),
  );
}

function buildMorningIndexRows(top5 = [], predictorTop5 = {}) {
  const rows = [];
  const seen = new Set();

  const pushRow = (item, fallbackScore = 0) => {
    const horseNo = String(item?.horse_no || "").trim();
    const horseName = String(item?.horse_name || "").trim();
    const dedupeKey = horseNo || horseName;
    if (!dedupeKey || seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    const sourceCount = Array.isArray(item?.sources) ? item.sources.length : 0;
    rows.push({
      horseNo: horseNo || "-",
      horseName: horseName || "-",
      aiIndex: Number(item?.support_score || fallbackScore || 0) || 0,
      score: Number(item?.support || 0) || 0,
      supportCount: sourceCount,
      mainCount: sourceCount,
    });
  };

  (Array.isArray(top5) ? top5 : []).filter(Boolean).forEach((item) => pushRow(item));

  if (rows.length < 5 && predictorTop5 && typeof predictorTop5 === "object") {
    for (const predictorId of ["main", "v6_kiwami"]) {
      for (const item of Array.isArray(predictorTop5?.[predictorId]) ? predictorTop5[predictorId] : []) {
        pushRow(item, 40);
        if (rows.length >= 5) break;
      }
      if (rows.length >= 5) break;
    }
  }

  return rows
    .sort((left, right) => right.aiIndex - left.aiIndex || right.score - left.score)
    .slice(0, 5);
}

function buildMorningCompareCards(predictorTop5, scheduledOffTime = "") {
  const source = predictorTop5 && typeof predictorTop5 === "object" ? predictorTop5 : {};
  const releaseText = buildFullReleaseText(scheduledOffTime);
  return PREDICTOR_ORDER
    .map((predictorId) => {
      const ranking = Array.isArray(source?.[predictorId]) ? source[predictorId].slice(0, 5) : [];
      if (!ranking.length) {
        return {
          predictor_id: predictorId,
          label: PREDICTOR_LABELS[predictorId] || predictorId,
          is_placeholder: true,
          placeholder_text: releaseText,
        };
      }
      const marksText = ranking
        .slice(0, MARK_ORDER.length)
        .map((item, index) => {
          const horseNo = String(item?.horse_no || "").trim();
          if (!horseNo) return "";
          return `${MARK_ORDER[index]}${horseNo}`;
        })
        .filter(Boolean)
        .join(" ");
      if (!marksText) {
        return {
          predictor_id: predictorId,
          label: PREDICTOR_LABELS[predictorId] || predictorId,
          is_placeholder: true,
          placeholder_text: "速報生成済み・馬番整備中",
        };
      }
      return {
        predictor_id: predictorId,
        label: PREDICTOR_LABELS[predictorId] || predictorId,
        marks_text: marksText,
      };
    })
    .filter(Boolean);
}

function mergeCompareCards(primaryCards = [], fallbackCards = []) {
  const fallbackMap = new Map(
    (Array.isArray(fallbackCards) ? fallbackCards : [])
      .filter(Boolean)
      .map((card) => [String(card?.predictor_id || "").trim(), card]),
  );
  const primaryMap = new Map(
    (Array.isArray(primaryCards) ? primaryCards : [])
      .filter(Boolean)
      .map((card) => [String(card?.predictor_id || "").trim(), card]),
  );

  return PREDICTOR_ORDER.map((predictorId) => {
    const primary = primaryMap.get(predictorId);
    if (primary) return primary;
    const fallback = fallbackMap.get(predictorId);
    if (fallback) return fallback;
    return null;
  }).filter(Boolean);
}

function buildConfidenceMeta(signalRows, totalSources) {
  const top = signalRows[0] || null;
  const next = signalRows[1] || null;
  if (!top || totalSources <= 0) {
    return { percentText: "-", supportText: "-", modelCountText: "0モデル", score: 0 };
  }

  const supportRatio = top.mainCount / totalSources;
  const marginRatio = top.score > 0
    ? Math.max(0, (top.score - Number(next?.score || 0)) / top.score)
    : 0;
  const score = Math.max(0, Math.min(1, 0.55 * supportRatio + 0.45 * marginRatio));
  return {
    percentText: `${Math.round(score * 100)}%`,
    supportText: `${top.mainCount}/${totalSources}モデル支持`,
    modelCountText: `${totalSources}モデル`,
    score,
  };
}

function DetailSummary({ label, value, accent = false }) {
  return (
    <article
      className={`race-detail-summary-card${accent ? " race-detail-summary-card--accent" : ""}`}
    >
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </article>
  );
}

function DetailAppSummary() {
  return (
    <article className="race-detail-summary-card race-detail-summary-card--app">
      <span>アプリ</span>
      <a
        className="race-detail-summary-card__app-link"
        href={APP_DOWNLOAD_HREF}
        aria-label="Androidアプリをダウンロード"
      >
        <img
          className="race-detail-summary-card__app-badge"
          src={APP_BADGE_SRC}
          alt="Google Play で手に入れよう"
        />
      </a>
    </article>
  );
}

function RaceSequenceLink({ label, emptyLabel, race, search }) {
  if (!race) {
    return (
      <article className="race-detail-race-link race-detail-race-link--disabled">
        <span>{label}</span>
        <strong>{emptyLabel}</strong>
      </article>
    );
  }

  const meta = raceNavigationMeta(race);
  return (
    <a
      className="race-detail-race-link"
      href={buildRaceDetailHref(race, search)}
      aria-label={`${label}へ移動`}
    >
      <span>{label}</span>
      <strong>{raceNavigationLabel(race)}</strong>
      {meta ? <small>{meta}</small> : null}
    </a>
  );
}

function RaceDetailNavigation({ navigation, search }) {
  if (!navigation?.hasNavigation) return null;

  return (
    <section className="race-detail-nav-panel" aria-label="レース移動">
      <div className="race-detail-race-switch" aria-label="同じ競馬場の前後R">
        <RaceSequenceLink
          label="前R"
          emptyLabel="前Rなし"
          race={navigation.previousRace}
          search={search}
        />
        <RaceSequenceLink
          label="次R"
          emptyLabel="次Rなし"
          race={navigation.nextRace}
          search={search}
        />
      </div>

      {navigation.courseGroups.length > 1 ? (
        <nav className="race-detail-course-switch" aria-label="競馬場切り替え">
          <span>競馬場</span>
          {navigation.courseGroups.map((group) => (
            <a
              key={group.location}
              className={group.active ? "is-active" : ""}
              href={buildRaceDetailHref(group.targetRace, search)}
              aria-current={group.active ? "page" : undefined}
              style={{
                "--course-accent": group.accent.accent,
                "--course-accent-soft": group.accent.soft,
              }}
            >
              <strong>{group.location}</strong>
              <em>{group.count}</em>
            </a>
          ))}
        </nav>
      ) : null}
    </section>
  );
}

function PanelEmpty({ children }) {
  return <p className="race-detail-empty-note">{children}</p>;
}

function cleanPredictionCopy(value, race = null) {
  const source = String(value || "").trim();
  if (source === "Mock LLM commentary") return "";
  const blockedTerms = [
    "heuristic scoring",
    "ルールベース評価",
    "正式なML model",
    "正式な機械学習モデル",
    "LLM simulation fallback used",
    "fallback",
    "oddsまたは人気が欠損",
    "データ処理",
    "内部実装",
  ];
  if (blockedTerms.some((term) => source.includes(term))) return "";
  const decisionLabel = race ? resolvePublicDecision(race).label : "注目";
  return source
    .trim()
    .replace(
      "上位比較で決め手が弱く、リスクまたはオッズ条件も不十分なため見送り。",
      "上位の能力差が詰まっており、軸を決め切るには決定打が足りないため見送り。",
    )
    .replace(/^本命は[^。]+。\s*/, "")
    .replace(/近走データ使用数=\d+、参考メモ使用数=\d+。\s*/g, "")
    .replaceAll("買い目", "予測印")
    .replaceAll("買い判断=BET", `判断=${decisionLabel}`)
    .replaceAll("判断=BET", `判断=${decisionLabel}`)
    .replaceAll("買い判断=SKIP", "判断=見送り")
    .replaceAll("判断=SKIP", "判断=見送り")
    .replaceAll("買い判断=NO_BET", "判断=見送り")
    .replaceAll("判断=NO_BET", "判断=見送り")
    .replaceAll("買い判断=", "判断=")
    .replaceAll("購入候補", decisionLabel)
    .replaceAll("候補なし", "印なし")
    .replaceAll("SKIP", "見送り")
    .replaceAll("BET", decisionLabel)
    .replaceAll("confidence=", "信頼度=")
    .replaceAll("信頼度=high", "信頼度=高")
    .replaceAll("信頼度=medium", "信頼度=中")
    .replaceAll("信頼度=low", "信頼度=低")
    .replaceAll("recent_runs使用数", "近走データ使用数")
    .replaceAll("lessons使用数", "参考メモ使用数")
    .replace(/近走データ使用数=\d+、参考メモ使用数=\d+。\s*/g, "")
    .replaceAll("lesson", "参考メモ")
    .replaceAll("過去参考メモ参考", "過去参考メモ")
    .replaceAll("unknown", "不明")
    .replaceAll("average", "平均ペース");
}

function buildAgentMarkRows(prediction, race = {}) {
  const topHorses = Array.isArray(prediction?.top_horses)
    ? prediction.top_horses.filter(Boolean)
    : [];
  const rows = topHorses.map((item, index) => ({
    mark: String(item?.mark || MARK_ORDER[index] || "").trim(),
    horseNo: String(item?.horse_no || item?.horseNo || "").trim(),
    horseName: String(item?.horse_name || item?.horseName || "").trim(),
    score: item?.total_score ?? item?.score ?? "",
  }));

  if (rows.length) {
    return rows.slice(0, 5);
  }

  return (Array.isArray(race?.top5) ? race.top5 : [])
    .filter(Boolean)
    .slice(0, 5)
    .map((item, index) => ({
      mark: MARK_ORDER[index] || "",
      horseNo: String(item?.horse_no || "").trim(),
      horseName: String(item?.horse_name || "").trim(),
      score: item?.support_score ?? item?.support ?? "",
    }));
}

function pickReasonBullets(prediction, race = null) {
  const strategyReason = cleanPredictionCopy(prediction?.strategy?.reason, race);
  const summary = cleanPredictionCopy(prediction?.summary, race);
  const risks = Array.isArray(prediction?.risks)
    ? prediction.risks.map((item) => cleanPredictionCopy(item, race)).filter(Boolean)
    : [];
  const bullets = [];
  if (strategyReason) bullets.push(strategyReason);
  if (summary) {
    const summaryParts = summary
      .split("。")
      .map((item) => item.trim())
      .filter(Boolean)
      .filter((item) => !item.startsWith("本命は"))
      .filter((item) => !item.includes("近走データ使用数") && !item.includes("参考メモ使用数"))
      .slice(0, 2);
    bullets.push(...summaryParts.map((item) => `${item}。`));
  }
  bullets.push(...risks.slice(0, 2));
  return bullets.slice(0, 4);
}

function AgentMarksPanel({ prediction, race }) {
  if (!prediction) return null;
  const markRows = buildAgentMarkRows(prediction, race);
  const reasons = pickReasonBullets(prediction, race);

  return (
    <section className="race-detail-panel race-detail-panel--marks-overview" id="race-detail-marks">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">AI予測</span>
          <h2>予測印</h2>
        </div>
      </div>

      <div className="race-detail-marks-overview">
        {markRows.length ? (
          <div className="race-detail-marks-list">
            {markRows.map((row) => (
              <article key={`${row.mark}-${row.horseNo}`} className={row.mark === MAIN_MARK ? "is-main" : ""}>
                <em>{row.mark || "-"}</em>
                <div>
                  <strong>{`${row.horseNo || "-"} ${row.horseName || ""}`.trim()}</strong>
                  {row.score !== "" && row.score !== null && row.score !== undefined ? (
                    <small>{`評価 ${row.score}`}</small>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <PanelEmpty>予測印はまだありません。</PanelEmpty>
        )}
      </div>

      <div className="race-detail-mark-reasons">
        <strong>判断理由</strong>
        {reasons.length ? (
          <ul>
            {reasons.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          <p>判断理由はまだありません。</p>
        )}
      </div>
    </section>
  );
}

function AgentHorseMemoPanel({ prediction }) {
  const topHorses = Array.isArray(prediction?.top_horses)
    ? prediction.top_horses.filter(Boolean).slice(0, 5)
    : [];
  if (!topHorses.length) return null;

  return (
    <section className="race-detail-panel race-detail-panel--agent-horses" id="race-detail-agent-horses">
      <div className="race-detail-panel__head">
        <div>
          <h2>上位馬メモ</h2>
        </div>
      </div>
      <div className="race-detail-agent-horse-list">
        {topHorses.map((item) => (
          <article key={`${item?.mark || "rank"}-${item?.horse_no}`} className="race-detail-agent-horse">
            <div className="race-detail-agent-horse__head">
              <span>{item?.mark || item?.pred_rank || "-"}</span>
              <strong>{`${item?.horse_no || "-"} ${item?.horse_name || "-"}`}</strong>
              <em>{item?.total_score ?? "-"}</em>
            </div>
            {item?.reason ? <p>{item.reason}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function MarkGrid({ marks, itemKey }) {
  return (
    <div className="race-detail-mark-grid">
      {MARK_ORDER.map((symbol) => {
        const horseNo = pickHorse(marks, symbol) || "-";
        return (
          <span
            key={`${itemKey}-${symbol}`}
            className={symbol === MAIN_MARK ? "is-main" : ""}
          >
            <em>{symbol}</em>
            <strong>{horseNo}</strong>
          </span>
        );
      })}
    </div>
  );
}

function AiIndexPanel({ signalRows }) {
  return (
    <section className="race-detail-panel" id="race-detail-index">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">AI指数</span>
          <h2>上位候補ランキング</h2>
        </div>
      </div>
      {signalRows.length ? (
        <div className="race-detail-index-list">
          {signalRows.slice(0, 5).map((item, index) => (
            <article key={item.horseNo} className="race-detail-index-item">
              <span>{item.horseNo}</span>
              <strong>{item.horseName || "-"}</strong>
              <em>{item.aiIndex}</em>
            </article>
          ))}
        </div>
      ) : (
        <PanelEmpty>AI指数を計算できるデータがまだありません。</PanelEmpty>
      )}
    </section>
  );
}

function RaceResultPanel({ resultEntries, resultText }) {
  return (
    <section className="race-detail-panel race-detail-panel--result" id="race-detail-result">
      <div className="race-detail-panel__head">
        <div>
          <span className="race-detail-panel__eyebrow">レース結果</span>
          <h2>確定着順</h2>
        </div>
      </div>
      {resultEntries.length ? (
        <ol className="race-detail-result-list">
          {resultEntries.map((entry) => (
            <li key={entry.key}>
              <span>{entry.rank}着</span>
              <strong>{entry.body}</strong>
            </li>
          ))}
        </ol>
      ) : (
        <p className="race-detail-result-placeholder">{resultText}</p>
      )}
    </section>
  );
}

function ConditionPredictorRankingPanel({ ranking }) {
  const cards = Array.isArray(ranking?.cards) ? ranking.cards : [];

  return (
    <section className="race-detail-panel race-detail-panel--condition-ranking" id="race-detail-condition-ranking">
      <div className="race-detail-panel__head">
        <div>
          <h2>この条件の定量モデル順位</h2>
        </div>
      </div>
      {ranking?.condition_text ? (
        <p className="race-detail-condition-ranking__meta">
          {ranking.condition_text}
          {ranking?.sample_count ? ` / 対象 ${ranking.sample_count}レース` : ""}
        </p>
      ) : null}
      {cards.length ? (
        <ol className="race-detail-condition-ranking__list">
          {cards.map((item) => (
            <li key={item.predictor_id} className="race-detail-condition-ranking__item">
              <span>{`${item.rank || "-"}`}</span>
              <strong>{item.label || "-"}</strong>
              <em>{item.top5_to_top3_hit_rate_text || "-"}</em>
            </li>
          ))}
        </ol>
      ) : (
        <PanelEmpty>この条件に一致する履歴データはまだありません。</PanelEmpty>
      )}
    </section>
  );
}

function PredictorCompareRow({ card }) {
  if (card?.is_placeholder) {
    return (
      <article className="race-detail-compare-row race-detail-compare-row--placeholder">
        <div className="race-detail-compare-row__top">
          <strong>{card?.label || "-"}</strong>
          <span className="race-detail-compare-status" aria-label="計算中">
            <span />
            <span />
            <span />
          </span>
        </div>
        <div className="race-detail-compare-loading" aria-hidden="true">
          <span className="race-detail-compare-loading__bar" />
        </div>
        <p className="race-detail-compare-placeholder">{card?.placeholder_text || "完全公開準備中"}</p>
      </article>
    );
  }
  const marks = parseMarks(card?.marks_text);

  return (
    <article className="race-detail-compare-row">
      <div className="race-detail-compare-row__top">
        <strong>{card?.label || "-"}</strong>
      </div>
      <MarkGrid
        marks={marks}
        itemKey={card?.predictor_id || card?.label || "predictor-compare"}
      />
    </article>
  );
}

export default function RaceDetailPage({ race, races = [], search = "", appShell = false }) {
  const predictorCompareCards = Array.isArray(race?.predictor_compare_cards)
    ? race.predictor_compare_cards.filter(Boolean)
    : [];
  const agentPrediction = race?.agent_prediction || null;
  const variant = String(race?.display_variant || "").trim();
  const isMorningPreview = variant === "morning_preview";
  const morningCompareCards = useMemo(
    () => buildMorningCompareCards(race?.predictor_top5, race?.scheduled_off_time),
    [race?.predictor_top5, race?.scheduled_off_time],
  );
  const compareCards = useMemo(
    () => mergeCompareCards(predictorCompareCards, morningCompareCards),
    [morningCompareCards, predictorCompareCards],
  );
  const status = race?.display_status || {};
  const isSettled = variant === "settled";
  const resultText = isSettled
    ? String(race?.display_body?.result_text || "結果はまだありません")
    : "結果は確定後に表示されます";
  const resultEntries = parseResultEntries(resultText);
  const signalRows = useMemo(
    () =>
      predictorCompareCards.length
        ? buildHorseSignalRows(predictorCompareCards)
        : buildMorningIndexRows(race?.top5, race?.predictor_top5),
    [predictorCompareCards, race?.predictor_top5, race?.top5],
  );
  const derivedSummary = useMemo(
    () =>
      !Array.isArray(race?.top5) || !race.top5.length
        ? buildPredictorConsensusSummary(race?.predictor_compare_cards)
        : null,
    [race?.predictor_compare_cards, race?.top5],
  );
  const badges = formatRaceBadges(race).filter(
    (item) => !["良", "稍重", "重", "不良"].includes(String(item || "").trim()),
  );
  const backHref = buildBackHref(search);
  const detailTitle = String(
    race?.display_header?.detail_title || race?.display_header?.title || "-",
  ).trim() || "-";
  const detailConfidenceText =
    Number.isFinite(Number(race?.confidence_score ?? derivedSummary?.confidence_score))
      ? `${Math.round(Number(race?.confidence_score ?? derivedSummary?.confidence_score) * 100)}%`
      : "-";
  const conditionRanking = race?.condition_predictor_ranking || {};
  const hasConditionRanking = Array.isArray(conditionRanking?.cards) && conditionRanking.cards.length > 0;
  const detailNavigation = useMemo(
    () => buildDetailNavigation(race, races),
    [race, races],
  );

  return (
    <section className="race-detail-page">
      <div className="race-detail-hero" id="race-detail-summary">
        <div className="race-detail-hero__copy">
          <a className="race-detail-back-link" href={backHref}>
            予測一覧に戻る
          </a>
          <span className="race-detail-hero__eyebrow">レース詳細</span>
          <AutoFitLine
            as="h1"
            className="race-detail-hero__title"
            maxFontSize={44}
            minFontSize={14}
          >
            {detailTitle}
          </AutoFitLine>
          {badges.length ? (
            <div className="race-detail-hero__badges">
              {badges.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          ) : null}
        </div>

        <div className="race-detail-hero__meta">
          <DetailSummary label="公開状態" value={status?.label || "公開中"} accent />
          <DetailSummary label="信頼度" value={detailConfidenceText} />
          <DetailAppSummary />
        </div>
      </div>

      <RaceDetailNavigation navigation={detailNavigation} search={search} />

      {agentPrediction ? (
        <AgentMarksPanel
          prediction={agentPrediction}
          race={race}
        />
      ) : null}

      {agentPrediction ? (
        <>
          <AgentHorseMemoPanel prediction={agentPrediction} />
          <RaceResultPanel resultEntries={resultEntries} resultText={resultText} />
        </>
      ) : (
        <div className="race-detail-layout">
          <AiIndexPanel signalRows={signalRows} />
          <div className="race-detail-side-stack">
            <RaceResultPanel resultEntries={resultEntries} resultText={resultText} />
          </div>
        </div>
      )}

      {!agentPrediction ? (
        <div className="race-detail-layout race-detail-layout--compare">
        <section
          className="race-detail-panel race-detail-panel--compare"
          id="race-detail-compare"
        >
          <div className="race-detail-panel__head">
            <div>
              <h2>
                {isMorningPreview && !predictorCompareCards.length
                  ? "定量モデル比較"
                  : "6モデルの本命比較"}
              </h2>
            </div>
          </div>
          <div className="race-detail-compare-list">
            {compareCards.length ? (
              compareCards.map((card) => (
                <PredictorCompareRow
                  key={card?.predictor_id || card?.label}
                  card={card}
                />
              ))
            ) : (
              <PanelEmpty>定量モデルの比較データはまだありません。</PanelEmpty>
            )}
          </div>
        </section>

        <div className="race-detail-side-stack">
          {hasConditionRanking ? (
            <ConditionPredictorRankingPanel ranking={conditionRanking} />
          ) : null}
        </div>
      </div>
      ) : null}
    </section>
  );
}
