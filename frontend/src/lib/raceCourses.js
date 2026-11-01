import catalog from "../fixtures/raceCourses.json" with { type: "json" };

export const COURSE_PROFILES = catalog.profiles;
export const COURSE_CHECKED_AT = catalog.checked_at;
export const TERRAIN_RESPONSE = Object.freeze({ speedPerGrade: 18, uphillEnergy: 16, downhillEnergy: 4 });
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const modulo = (x, n) => ((x % n) + n) % n;

export function resolveRaceCourse(info, requestedId = null) {
  if (requestedId) {
    const selected = COURSE_PROFILES.find((p) => p.id === requestedId);
    if (!selected) throw new Error(`Unknown course profile: ${requestedId}`);
    return { course: selected, ambiguous: false, comparison: selected.course !== info.course || selected.surface !== info.surface };
  }
  const candidates = COURSE_PROFILES.filter((p) => p.course === String(info.course || "").replace("競馬場", "") && p.surface === info.surface && p.distances_m.includes(info.distance));
  const explicit = info.course_layout || info.layout;
  const matched = explicit ? candidates.find((p) => p.layout === explicit) : null;
  // An unresolved inner/outer choice is kept visible; never average two tracks.
  const selected = matched || candidates.find((p) => p.layout === "outer") || candidates[0] || null;
  return { course: selected, ambiguous: !matched && candidates.length > 1, comparison: false };
}

function elevationAtRemaining(course, remaining) {
  const points = course?.elevation_points;
  if (!points?.length) return null;
  const lap = course.lap_m || 1000;
  const r = course.direction === "straight" ? clamp(remaining, 0, 1000) : modulo(remaining, lap);
  let lo = 0, hi = points.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (points[mid][0] <= r) lo = mid; else hi = mid - 1;
  }
  const a = points[lo], b = points[Math.min(lo + 1, points.length - 1)];
  return b[0] === a[0] ? a[1] : a[1] + (b[1] - a[1]) * (r - a[0]) / (b[0] - a[0]);
}

export function terrainAt(course, distance, raceDistance) {
  if (!course?.elevation_points) return { known: false, height: null, grade: 0, label: "坂位置未設定" };
  const remaining = Math.max(0, raceDistance - distance);
  const height = elevationAtRemaining(course, remaining);
  const forward = course.direction === "straight" ? Math.max(0, remaining - 10) : remaining - 10;
  const backward = course.direction === "straight" ? Math.min(1000, remaining + 10) : remaining + 10;
  const grade = clamp((elevationAtRemaining(course, forward) - elevationAtRemaining(course, backward)) / Math.max(1, backward - forward), -0.05, 0.05);
  return { known: true, height, grade, label: grade > 0.0015 ? "上り" : grade < -0.0015 ? "下り" : "ほぼ平坦" };
}

export function terrainResponse(terrain, enabled = true) {
  if (!enabled || !terrain.known) return { speed: 0, energy: 1 };
  return {
    speed: -terrain.grade * TERRAIN_RESPONSE.speedPerGrade,
    energy: clamp(1 + Math.max(0, terrain.grade) * TERRAIN_RESPONSE.uphillEnergy + Math.min(0, terrain.grade) * TERRAIN_RESPONSE.downhillEnergy, 0.85, 1.8),
  };
}

const paths = new Map();
function coursePath(course) {
  if (paths.has(course.id)) return paths.get(course.id);
  const control = course.plan_points;
  const points = [];
  if (course.direction === "straight") points.push(...control.map(([x,y]) => ({ x,y })));
  else {
    for (let i = 0; i < control.length; i += 1) {
      const a = control[modulo(i - 1, control.length)], b = control[i], c = control[(i + 1) % control.length], d = control[(i + 2) % control.length];
      for (let j = 0; j < 18; j += 1) {
        const t = j / 18;
        const values = [0,1].map((axis) => 0.5 * ((2*b[axis]) + (-a[axis]+c[axis])*t + (2*a[axis]-5*b[axis]+4*c[axis]-d[axis])*t*t + (-a[axis]+3*b[axis]-3*c[axis]+d[axis])*t*t*t));
        points.push({ x: values[0], y: values[1] });
      }
    }
    points.push({ ...points[0] });
  }
  let length = 0;
  points.forEach((p,i) => { if (i) length += Math.hypot(p.x-points[i-1].x,p.y-points[i-1].y); p.s=length; });
  const result = { points, length };
  paths.set(course.id, result);
  return result;
}

function pointOnPath(course, fraction) {
  const { points, length } = coursePath(course);
  const s = clamp(fraction,0,1)*length;
  let lo=0,hi=points.length-1;
  while (lo<hi) { const mid=Math.ceil((lo+hi)/2); if(points[mid].s<=s)lo=mid;else hi=mid-1; }
  const a=points[lo],b=points[Math.min(lo+1,points.length-1)];
  const u=b.s===a.s?0:(s-a.s)/(b.s-a.s);
  return { x:a.x+(b.x-a.x)*u, y:a.y+(b.y-a.y)*u, angle:Math.atan2(b.y-a.y,b.x-a.x) };
}

export function coursePoint(course, distance, raceDistance, lane = 0, elevationScale = 5) {
  if (!course) return null;
  const fraction = course.direction === "straight" ? clamp(distance/raceDistance,0,1) : modulo(distance-raceDistance,course.lap_m)/course.lap_m;
  const point=pointOnPath(course,fraction);
  const terrain=terrainAt(course,distance,raceDistance);
  const y=point.y-(terrain.height || 0)*elevationScale;
  const offset=(lane-3)*2.3;
  return { x:point.x-Math.sin(point.angle)*offset, y:y+Math.cos(point.angle)*offset, angle:point.angle };
}

export function courseSection(course, distance, raceDistance) {
  if (!course) return { label:"走路未設定",isBend:false };
  if (course.direction === "straight") return { label:"直線コース",isBend:false };
  const remaining=modulo(raceDistance-distance,course.lap_m);
  if (remaining <= course.straight_m) return { label:raceDistance-distance<=course.straight_m?"最後の直線":"正面の直線",isBend:false };
  const fraction=modulo(distance-raceDistance,course.lap_m)/course.lap_m;
  const a=pointOnPath(course,modulo(fraction-.012,1)),b=pointOnPath(course,modulo(fraction+.012,1));
  const turn=Math.atan2(Math.sin(b.angle-a.angle),Math.cos(b.angle-a.angle));
  const isBend=Math.abs(turn)>.11;
  return { label:isBend?"コーナー":"道中の直線",isBend };
}

export function courseGeometryPath(course, raceDistance, elevationScale = 5) {
  const length=course.direction==="straight"?1000:course.lap_m;
  return Array.from({length:181},(_,i)=> {
    const point=coursePoint(course,length*i/180,length,3,elevationScale);
    return `${i?"L":"M"}${point.x.toFixed(2)},${point.y.toFixed(2)}`;
  }).join(" ");
}

export function raceElevationSamples(course, raceDistance, count = 160) {
  return Array.from({length:count+1},(_,i)=>({ distance:raceDistance*i/count,...terrainAt(course,raceDistance*i/count,raceDistance) }));
}
