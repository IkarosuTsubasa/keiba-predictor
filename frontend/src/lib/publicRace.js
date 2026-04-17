export const APP_BASE_PATH = "/keiba";
export const MARK_ORDER = ["◎", "○", "▲", "△", "☆"];
const SUMMARY_MARK_WEIGHTS = {
  "◎": 1.0,
  "○": 0.78,
  "▲": 0.62,
  "△": 0.48,
  "☆": 0.36,
};

export function parseMarks(text) {
  return [...String(text || "").matchAll(/([◎○▲△☆])\s*([0-9]+)/g)].map(
    (item) => ({
      symbol: item[1],
      horseNo: item[2],
    }),
  );
}

export function pickHorse(marks, symbol) {
  const found = (marks || []).find((item) => item.symbol === symbol);
  return found ? found.horseNo : null;
}

function numericHorseNo(value) {
  const number = Number(String(value || "").trim());
  return Number.isFinite(number) ? number : Number.MAX_SAFE_INTEGER;
}

export function buildPredictorConsensusSummary(cards) {
  const validCards = Array.isArray(cards) ? cards.filter(Boolean) : [];
  if (!validCards.length) {
    return null;
  }

  const horseScores = new Map();
  const topPickCounts = new Map();

  for (const card of validCards) {
    const marks = parseMarks(card?.marks_text);
    if (!marks.length) continue;

    for (const item of marks) {
      const horseNo = String(item?.horseNo || "").trim();
      const weight = SUMMARY_MARK_WEIGHTS[item?.symbol] || 0;
      if (!horseNo || weight <= 0) continue;
      horseScores.set(horseNo, (horseScores.get(horseNo) || 0) + weight);
      if (item.symbol === "◎") {
        topPickCounts.set(horseNo, (topPickCounts.get(horseNo) || 0) + 1);
      }
    }
  }

  const ranked = [...horseScores.entries()]
    .map(([horseNo, score]) => ({
      horse_no: horseNo,
      support: score,
      support_score: Math.max(1, Math.min(99, Math.round(48 + score * 12))),
      top_pick_count: topPickCounts.get(horseNo) || 0,
    }))
    .sort((left, right) => {
      if (right.support !== left.support) return right.support - left.support;
      if (right.top_pick_count !== left.top_pick_count) {
        return right.top_pick_count - left.top_pick_count;
      }
      return numericHorseNo(left.horse_no) - numericHorseNo(right.horse_no);
    });

  if (!ranked.length) {
    return null;
  }

  const top5 = ranked.slice(0, 5);
  const modelCount = validCards.length;
  const lead = top5[0];
  const second = top5[1];
  const totalSupport = top5.reduce((sum, item) => sum + item.support, 0);
  const agreementScore = modelCount > 0 ? lead.top_pick_count / modelCount : 0;
  const concentrationScore = totalSupport > 0 ? lead.support / totalSupport : 0;
  const marginScore =
    lead && second && lead.support > 0
      ? Math.max(0, (lead.support - second.support) / lead.support)
      : 0.32;
  const coverageScore = Math.min(1, modelCount / 6);
  const confidenceScore = Math.max(
    0.12,
    Math.min(
      0.97,
      0.42 * agreementScore +
        0.28 * concentrationScore +
        0.18 * marginScore +
        0.12 * coverageScore,
    ),
  );

  return {
    model_count: modelCount,
    confidence_score: Number(confidenceScore.toFixed(6)),
    agreement_score: Number(agreementScore.toFixed(6)),
    top5,
  };
}

export function raceIdentifier(race) {
  return String(race?.run_id || race?.card_id || race?.race_id || "").trim();
}

export function buildRaceDetailHref(race, search = "") {
  const id = raceIdentifier(race);
  const query = String(search || "").replace(/^\?/, "");
  if (!id) {
    return query ? `${APP_BASE_PATH}?${query}` : APP_BASE_PATH;
  }
  return query
    ? `${APP_BASE_PATH}/race/${encodeURIComponent(id)}?${query}`
    : `${APP_BASE_PATH}/race/${encodeURIComponent(id)}`;
}

export function matchRaceIdentifier(race, targetId) {
  const normalizedTarget = String(targetId || "").trim();
  if (!normalizedTarget) return false;

  return [race?.run_id, race?.card_id, race?.race_id, ...(Array.isArray(race?.alias_ids) ? race.alias_ids : [])]
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .includes(normalizedTarget);
}

export function parseResultEntries(text) {
  const source = String(text || "").trim();
  if (!source || source.includes("未") || source.includes("待ち")) {
    return [];
  }

  return source
    .split("/")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((item, index) => {
      const match = item.match(/^([1-3])着\s*(.+)$/);
      const rank = Number(match?.[1] || index + 1);
      const body = String(match?.[2] || item).trim();
      return { key: `${rank}-${body}`, rank, body };
    })
    .filter((item) => item.body);
}

export function isTrackConditionBadge(value) {
  return ["良", "稍重", "重", "不良"].includes(String(value || "").trim());
}

export function formatRaceBadges(race) {
  const badges = Array.isArray(race?.display_header?.badges)
    ? race.display_header.badges
    : [];
  return badges.filter(Boolean);
}
