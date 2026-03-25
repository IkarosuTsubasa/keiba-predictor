export const APP_BASE_PATH = "/keiba";
export const MARK_ORDER = ["◎", "○", "▲", "△", "☆"];

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

  return [race?.run_id, race?.card_id, race?.race_id]
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
