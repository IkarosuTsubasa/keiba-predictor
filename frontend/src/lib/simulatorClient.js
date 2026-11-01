const API = "/keiba/api/public/simulator";
export const ACTIVE_MODEL = "illustrative-terrain-v2";
export const BRIDGE_VERSION = 1;
export const STATUS_TEXT = {
  preparing: "データ準備中", not_found: "このレースのデータはまだありません。",
  version_not_found: "保存したデータが見つかりません。最新のデータを選んでください。",
  unsupported: "このレースはシミュレーションの対象外です。",
  unavailable: "出走予定の馬が足りないため、試走できません。",
  disabled: "シミュレーションは現在ご利用いただけません。",
};

export class SimulatorError extends Error {
  constructor(code) { super(STATUS_TEXT[code] || "データを取得できませんでした。通信状況を確認して、もう一度お試しください。"); this.code = code; }
}
export function validContext(context) {
  const stamp = Date.parse(context.date);
  if (!Number.isFinite(stamp) || new Date(stamp).toISOString().slice(0, 10) !== context.date) return false;
  return /^\d{12}$/.test(context.raceId || "") && /^\d{4}-\d{2}-\d{2}$/.test(context.date || "")
    && (!context.inputVersion || /^[a-f0-9]{24}$/.test(context.inputVersion))
    && (!context.modelVersion || context.modelVersion === ACTIVE_MODEL)
    && (context.seed == null || Number.isInteger(context.seed) && context.seed >= 0 && context.seed <= 0xffffffff);
}
export function contextFromUrl(url = window.location.href) {
  const parsed = new URL(url), q = parsed.searchParams;
  const pathId = parsed.pathname.match(/\/open\/simulator\/(\d{12})\/?$/)?.[1];
  return { raceId: pathId || q.get("race_id") || "", date: q.get("date") || "", inputVersion: q.get("iv") || "", modelVersion: q.get("mv") || ACTIVE_MODEL, seed: q.has("seed") ? Number(q.get("seed")) : null, source: ["web_detail", "app_tab", "app_race_detail"].includes(q.get("source")) ? q.get("source") : "web_detail" };
}
export async function fetchSimulation(context, signal, { latest = false } = {}) {
  if (!validContext(context)) throw new SimulatorError("invalid_context");
  const query = new URLSearchParams({ date: context.date });
  if (context.inputVersion && !latest) query.set("input_version", context.inputVersion);
  const result = await fetch(`${API}/races/${context.raceId}?${query}`, { signal });
  const payload = await result.json();
  if (!result.ok || !payload.ok) throw new SimulatorError(payload.error);
  const race = payload.data;
  if (race.race_id !== context.raceId || race.race_date !== context.date || race.race_info?.race_id !== context.raceId
    || race.active_model_version !== ACTIVE_MODEL || context.inputVersion && !latest && race.input_version !== context.inputVersion) throw new SimulatorError("identity_mismatch");
  return race;
}
export async function fetchSimulatorConfig(signal) {
  const result = await fetch(`${API}/models`, { signal });
  if (!result.ok) throw new SimulatorError("network_error");
  return (await result.json()).data;
}
export function appLink(context) {
  const q = new URLSearchParams({ date: context.date, mv: context.modelVersion || ACTIVE_MODEL, source: context.source || "web_detail" });
  if (context.inputVersion) q.set("iv", context.inputVersion);
  if (context.seed != null) q.set("seed", context.seed);
  return `/keiba/open/simulator/${context.raceId}?${q}`;
}
export function engineRace(payload) {
  return { ...payload, horses: payload.horses.filter(h => h.entry_status === "active") };
}

export function validateSettings(value = {}) {
  const out = {};
  if (value.pace != null) { if (!["slow", "average", "fast"].includes(value.pace)) throw new Error("invalid_pace"); out.pace = value.pace; }
  if (value.variation != null) { if (![0.4, 1, 1.7].includes(value.variation)) throw new Error("invalid_variation"); out.variation = value.variation; }
  if (value.speed != null) { if (![1, 2, 4, 8].includes(value.speed)) throw new Error("invalid_speed"); out.speed = value.speed; }
  if (value.view != null) { if (!["course", "pack"].includes(value.view)) throw new Error("invalid_view"); out.view = value.view; }
  if (value.seed != null) { if (!Number.isInteger(value.seed) || value.seed < 0 || value.seed > 0xffffffff) throw new Error("invalid_seed"); out.seed = value.seed; }
  if (Object.hasOwn(value, "distance")) { if (value.distance !== null && (!Number.isInteger(value.distance) || value.distance < 800 || value.distance > 4000)) throw new Error("invalid_distance"); out.distance = value.distance; }
  if (Object.hasOwn(value, "courseProfileId")) { if (value.courseProfileId !== null && typeof value.courseProfileId !== "string") throw new Error("invalid_course"); out.courseProfileId = value.courseProfileId; }
  return out;
}
