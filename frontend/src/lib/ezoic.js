const SLOT_IDS = {
  homeBeforeBoard: String(import.meta.env.VITE_EZOIC_SLOT_HOME_BEFORE_BOARD || "").trim(),
  homeAfterBoard: String(import.meta.env.VITE_EZOIC_SLOT_HOME_AFTER_BOARD || "").trim(),
  historyBetweenPanels: String(import.meta.env.VITE_EZOIC_SLOT_HISTORY_BETWEEN_PANELS || "").trim(),
  raceDetailSidebar: String(import.meta.env.VITE_EZOIC_SLOT_RACE_DETAIL_SIDEBAR || "").trim(),
  reportsBetweenPanels: String(import.meta.env.VITE_EZOIC_SLOT_REPORTS_BETWEEN_PANELS || "").trim(),
  reportDetailBody: String(import.meta.env.VITE_EZOIC_SLOT_REPORT_DETAIL_BODY || "").trim(),
};

let flushTimer = 0;
const pendingSlotIds = new Set();

function scheduleFlush() {
  if (typeof window === "undefined" || flushTimer) {
    return;
  }

  flushTimer = window.setTimeout(() => {
    flushTimer = 0;
    const slotIds = [...pendingSlotIds];
    pendingSlotIds.clear();
    if (!slotIds.length) {
      return;
    }

    const standalone = (window.ezstandalone = window.ezstandalone || {});
    standalone.cmd = standalone.cmd || [];
    standalone.cmd.push(() => {
      if (typeof standalone.showAds !== "function") {
        return;
      }
      if (slotIds.length === 1) {
        standalone.showAds(slotIds[0]);
        return;
      }
      standalone.showAds(slotIds);
    });
  }, 0);
}

export function getEzoicSlotId(slotName) {
  return SLOT_IDS[slotName] || "";
}

export function requestEzoicAds(slotIds = []) {
  for (const slotId of slotIds) {
    const id = String(slotId || "").trim();
    if (!id) continue;
    pendingSlotIds.add(id);
  }
  scheduleFlush();
}
