const COURSE_ACCENTS = [
  { accent: "#2563eb", soft: "#eff6ff" },
  { accent: "#4338ca", soft: "#eef2ff" },
  { accent: "#b45309", soft: "#fff7ed" },
  { accent: "#7c3aed", soft: "#f5f3ff" },
  { accent: "#be123c", soft: "#fff1f2" },
  { accent: "#0369a1", soft: "#f0f9ff" },
  { accent: "#475569", soft: "#f8fafc" },
];

const COURSE_ACCENT_BY_LOCATION = {
  川崎: { accent: "#0b3d91", soft: "#eff6ff" },
  名古屋: { accent: "#4338ca", soft: "#eef2ff" },
  園田: { accent: "#b45309", soft: "#fff7ed" },
  門別: { accent: "#2563eb", soft: "#eff6ff" },
  水沢: { accent: "#7c3aed", soft: "#f5f3ff" },
  大井: { accent: "#475569", soft: "#f8fafc" },
  金沢: { accent: "#b45309", soft: "#fff7ed" },
  東京: { accent: "#1d4ed8", soft: "#eff6ff" },
  京都: { accent: "#b45309", soft: "#fff7ed" },
  阪神: { accent: "#7c3aed", soft: "#f5f3ff" },
  中山: { accent: "#be123c", soft: "#fff1f2" },
  中京: { accent: "#0369a1", soft: "#f0f9ff" },
  札幌: { accent: "#4338ca", soft: "#eef2ff" },
  函館: { accent: "#475569", soft: "#f8fafc" },
  新潟: { accent: "#0284c7", soft: "#f0f9ff" },
  福島: { accent: "#c2410c", soft: "#fff7ed" },
  小倉: { accent: "#a21caf", soft: "#fdf4ff" },
  佐賀: { accent: "#7c3aed", soft: "#f5f3ff" },
  高知: { accent: "#be123c", soft: "#fff1f2" },
  船橋: { accent: "#475569", soft: "#f8fafc" },
};

export function resolveCourseAccent(location, index = 0) {
  return COURSE_ACCENT_BY_LOCATION[location] || COURSE_ACCENTS[index % COURSE_ACCENTS.length];
}
