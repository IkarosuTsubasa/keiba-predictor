export const HIGH_CONFIDENCE_THRESHOLD = 0.85;
export const MEDIUM_CONFIDENCE_THRESHOLD = 0.70;

const DECISION_LABELS = {
  high: "高評価",
  medium: "要確認",
  low: "見送り",
  pending: "確認待ち",
};

function safeText(value) {
  return String(value || "").trim();
}

export function confidenceBand(value) {
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) return "";
  if (confidence >= HIGH_CONFIDENCE_THRESHOLD) return "high";
  if (confidence >= MEDIUM_CONFIDENCE_THRESHOLD) return "medium";
  return "low";
}

export function resolvePublicDecision(race) {
  const explicitDecision = safeText(race?.agent_prediction?.strategy?.bet_decision).toUpperCase();
  if (explicitDecision === "SKIP" || explicitDecision === "NO_BET") {
    return { label: DECISION_LABELS.low, tone: "skip" };
  }

  const band = confidenceBand(race?.confidence_score);
  if (band === "high") return { label: DECISION_LABELS.high, tone: "bet" };
  if (explicitDecision === "BET" && band) return { label: DECISION_LABELS.medium, tone: "watch" };
  if (band === "medium") return { label: DECISION_LABELS.medium, tone: "watch" };
  if (band === "low") return { label: DECISION_LABELS.low, tone: "skip" };

  if (explicitDecision === "BET") return { label: DECISION_LABELS.high, tone: "bet" };

  const metaValue = safeText(race?.predictor_compare_cards?.[0]?.metaValue).toLowerCase();
  if (metaValue === "high") return { label: DECISION_LABELS.high, tone: "bet" };
  if (metaValue === "medium") return { label: DECISION_LABELS.medium, tone: "watch" };
  if (metaValue === "low") return { label: DECISION_LABELS.low, tone: "skip" };
  return { label: DECISION_LABELS.pending, tone: "watch" };
}
