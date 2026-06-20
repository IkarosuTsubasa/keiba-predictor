import React, {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import RaceCard from "./RaceCard";

const HIGH_CONFIDENCE_THRESHOLD = 0.85;
const MEDIUM_CONFIDENCE_THRESHOLD = 0.70;

function displayOrderValue(race) {
  const value = Number(race?.display_order);
  return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER;
}

function raceKey(race) {
  return `${race?.card_id || race?.run_id || race?.race_id || ""}-${race?.display_order ?? ""}`;
}

function raceLocation(race) {
  return String(race?.location || "").trim();
}

function raceTitle(race) {
  return String(race?.race_title || race?.display_header?.title || "").trim();
}

export function sortRacesForDisplay(races) {
  return [...(races || [])].sort(
    (a, b) => displayOrderValue(a) - displayOrderValue(b),
  );
}

function displayVariant(race) {
  return String(race?.display_variant || "").trim();
}

function isSettledRace(race) {
  return displayVariant(race) === "settled";
}

function isHighEvaluation(race) {
  const decision = String(race?.agent_prediction?.strategy?.bet_decision || "").trim();

  const metaValue = String(race?.predictor_compare_cards?.[0]?.metaValue || "").trim();
  const confidence = Number(race?.confidence_score);
  if (Number.isFinite(confidence)) return confidence >= HIGH_CONFIDENCE_THRESHOLD;
  if (decision === "BET") return true;
  return metaValue === "high";
}

function isSkipEvaluation(race) {
  const decision = String(race?.agent_prediction?.strategy?.bet_decision || "").trim();
  if (decision === "SKIP") return true;
  if (decision === "BET") return false;

  const metaValue = String(race?.predictor_compare_cards?.[0]?.metaValue || "").trim();
  const confidence = Number(race?.confidence_score);
  if (Number.isFinite(confidence)) return confidence < MEDIUM_CONFIDENCE_THRESHOLD;
  return metaValue === "low";
}

function filterRaces(races, statusFilter, locationFilter) {
  let filtered = races || [];
  if (statusFilter === "settled") {
    filtered = filtered.filter(isSettledRace);
  } else if (statusFilter === "open") {
    filtered = filtered.filter((race) => !isSettledRace(race));
  }
  if (locationFilter !== "all") {
    filtered = filtered.filter((race) => raceLocation(race) === locationFilter);
  }
  return filtered;
}

const STATUS_TABS = [
  { key: "all", label: "すべて" },
  { key: "open", label: "発売中" },
  { key: "settled", label: "確定済み" },
];

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

function courseAccent(location, index) {
  return COURSE_ACCENT_BY_LOCATION[location] || COURSE_ACCENTS[index % COURSE_ACCENTS.length];
}

function buildCourseGroups(races) {
  const groups = [];
  const indexByLocation = new Map();

  for (const race of races || []) {
    const location = raceLocation(race) || "その他";
    let groupIndex = indexByLocation.get(location);
    if (groupIndex === undefined) {
      groupIndex = groups.length;
      indexByLocation.set(location, groupIndex);
      groups.push({ location, races: [] });
    }
    groups[groupIndex].races.push(race);
  }

  return groups.map((group, index) => {
    const settledCount = group.races.filter(isSettledRace).length;
    return {
      ...group,
      key: `${index}-${group.location}`,
      firstRaceTitle: raceTitle(group.races[0]),
      accent: courseAccent(group.location, index),
      summary: {
        total: group.races.length,
        high: group.races.filter(isHighEvaluation).length,
        skip: group.races.filter(isSkipEvaluation).length,
        settled: settledCount,
        open: group.races.length - settledCount,
      },
    };
  });
}

function CourseGroupHeader({ group }) {
  const { summary } = group;
  const chips = [
    `${summary.total}レース`,
    `高評価 ${summary.high}`,
    `見送り ${summary.skip}`,
  ];

  if (summary.settled > 0) {
    chips.push(`結果確定 ${summary.settled}`);
  } else if (summary.open > 0) {
    chips.push(`発売中 ${summary.open}`);
  }

  return (
    <header className="race-course-group__header">
      <div className="race-course-group__title">
        <span className="race-course-group__eyebrow">競馬場</span>
        <strong>{group.location}</strong>
        {group.firstRaceTitle ? <small>{group.firstRaceTitle}から</small> : null}
      </div>
      <div className="race-course-group__summary" aria-label={`${group.location}の表示概要`}>
        {chips.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>
    </header>
  );
}

export default function RaceGrid({ races, appShell = false, onVisibleRacesChange = null }) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [locationFilter, setLocationFilter] = useState("all");
  const [viewportPhase, setViewportPhase] = useState("idle");
  const [indicator, setIndicator] = useState({ left: 0, width: 0, opacity: 0 });
  const sectionRef = useRef(null);
  const tabsRef = useRef(null);
  const buttonRefs = useRef(new Map());
  const itemRefs = useRef(new Map());
  const groupRefs = useRef(new Map());
  const rectsRef = useRef(new Map());
  const fadeOutTimerRef = useRef(0);
  const fadeInTimerRef = useRef(0);

  const locationTabs = useMemo(() => {
    const seen = new Set();
    const items = [];
    for (const race of sortRacesForDisplay(races || [])) {
      const location = raceLocation(race);
      if (!location || seen.has(location)) continue;
      seen.add(location);
      items.push(location);
    }
    return items;
  }, [races]);

  const filtered = useMemo(
    () => sortRacesForDisplay(filterRaces(races, statusFilter, locationFilter)),
    [races, statusFilter, locationFilter],
  );

  const courseGroups = useMemo(() => buildCourseGroups(filtered), [filtered]);

  useEffect(() => {
    if (typeof onVisibleRacesChange === "function") {
      onVisibleRacesChange(filtered);
    }
  }, [filtered, onVisibleRacesChange]);

  useEffect(() => {
    if (locationFilter === "all") return;
    if (!locationTabs.includes(locationFilter)) {
      setLocationFilter("all");
    }
  }, [locationFilter, locationTabs]);

  useEffect(() => {
    return () => {
      window.clearTimeout(fadeOutTimerRef.current);
      window.clearTimeout(fadeInTimerRef.current);
    };
  }, []);

  useLayoutEffect(() => {
    const container = tabsRef.current;
    const active = buttonRefs.current.get(statusFilter);
    if (!container || !active) {
      setIndicator((current) => ({ ...current, opacity: 0 }));
      return;
    }
    const containerRect = container.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    setIndicator({
      left: activeRect.left - containerRect.left,
      width: activeRect.width,
      opacity: 1,
    });
  }, [filtered.length, statusFilter]);

  useLayoutEffect(() => {
    const nextRects = new Map();
    for (const race of filtered) {
      const key = raceKey(race);
      const node = itemRefs.current.get(key);
      if (!node) continue;
      const rect = node.getBoundingClientRect();
      nextRects.set(key, rect);
      const previousRect = rectsRef.current.get(key);

      if (previousRect) {
        const deltaX = previousRect.left - rect.left;
        const deltaY = previousRect.top - rect.top;
        if (deltaX || deltaY) {
          node.animate(
            [
              { transform: `translate(${deltaX}px, ${deltaY}px)` },
              { transform: "translate(0, 0)" },
            ],
            {
              duration: 320,
              easing: "cubic-bezier(0.22, 1, 0.36, 1)",
            },
          );
        }
      } else {
        node.animate(
          [
            { opacity: 0, transform: "translateY(14px) scale(0.985)" },
            { opacity: 1, transform: "translateY(0) scale(1)" },
          ],
          {
            duration: 220,
            easing: "cubic-bezier(0.22, 1, 0.36, 1)",
          },
        );
      }
    }
    rectsRef.current = nextRects;
  }, [filtered]);

  const queueFilterChange = (update) => {
    window.clearTimeout(fadeOutTimerRef.current);
    window.clearTimeout(fadeInTimerRef.current);
    setViewportPhase("out");
    fadeOutTimerRef.current = window.setTimeout(() => {
      startTransition(update);
      setViewportPhase("in");
      fadeInTimerRef.current = window.setTimeout(() => {
        setViewportPhase("idle");
      }, 180);
    }, 100);
  };

  const changeStatusFilter = (nextStatus) => {
    if (nextStatus === statusFilter) return;
    queueFilterChange(() => setStatusFilter(nextStatus));
  };

  const changeLocationFilter = (nextLocation) => {
    if (nextLocation === locationFilter) return;
    queueFilterChange(() => setLocationFilter(nextLocation));
  };

  const scrollToSectionTop = () => {
    sectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const scrollToCourse = (location) => {
    const node = groupRefs.current.get(location);
    if (!node) return;
    node.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="race-grid-section" ref={sectionRef}>
      {!appShell ? (
        <>
          <div className="race-grid-controls">
            <div className="race-grid-tabs" ref={tabsRef} aria-label="公開状態">
              <span
                className="race-grid-tabs__indicator"
                aria-hidden="true"
                style={{
                  width: `${indicator.width}px`,
                  transform: `translateX(${indicator.left}px)`,
                  opacity: indicator.opacity,
                }}
              />
              {STATUS_TABS.map((item) => (
                <button
                  key={item.key}
                  ref={(node) => {
                    if (node) {
                      buttonRefs.current.set(item.key, node);
                    } else {
                      buttonRefs.current.delete(item.key);
                    }
                  }}
                  type="button"
                  className={statusFilter === item.key ? "is-active" : ""}
                  onClick={() => changeStatusFilter(item.key)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {locationTabs.length ? (
              <label className="race-grid-course-filter">
                <span>競馬場</span>
                <select
                  value={locationFilter}
                  onChange={(event) => changeLocationFilter(event.target.value)}
                >
                  <option value="all">すべての競馬場</option>
                  {locationTabs.map((location) => (
                    <option key={location} value={location}>
                      {location}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <span className="race-grid-filter-summary">
              表示 {filtered.length}レース
            </span>
          </div>

          {locationFilter === "all" && courseGroups.length > 1 ? (
            <nav className="race-course-nav" aria-label="競馬場ナビゲーション">
              <span>競馬場</span>
              <button type="button" onClick={scrollToSectionTop}>
                すべて
                <em>{filtered.length}</em>
              </button>
              {courseGroups.map((group) => (
                <button
                  key={group.key}
                  type="button"
                  className="race-course-nav__item"
                  onClick={() => scrollToCourse(group.location)}
                  style={{
                    "--course-accent": group.accent.accent,
                    "--course-accent-soft": group.accent.soft,
                  }}
                >
                  {group.location}
                  <em>{group.summary.total}</em>
                </button>
              ))}
            </nav>
          ) : null}
        </>
      ) : null}

      <div className={`race-grid__viewport race-grid__viewport--${viewportPhase}`}>
        {!appShell ? (
          <div className="race-grid-table-head" aria-hidden="true">
            <span>レース</span>
            <span>判断</span>
            <span>信頼度</span>
            <span>本命</span>
            <span>上位印</span>
            <span>結果</span>
            <span>詳細</span>
          </div>
        ) : null}
        <div className="race-course-groups">
          {courseGroups.map((group) => (
            <section
              key={group.key}
              ref={(node) => {
                if (node) {
                  groupRefs.current.set(group.location, node);
                } else {
                  groupRefs.current.delete(group.location);
                }
              }}
              className="race-course-group"
              style={{
                "--course-accent": group.accent.accent,
                "--course-accent-soft": group.accent.soft,
              }}
            >
              <CourseGroupHeader group={group} />
              <div className="race-grid race-grid--course">
                {group.races.map((race) => {
                  const key = raceKey(race);
                  return (
                    <div
                      key={key}
                      ref={(node) => {
                        if (node) {
                          itemRefs.current.set(key, node);
                        } else {
                          itemRefs.current.delete(key);
                        }
                      }}
                      className="race-grid__item"
                      style={{ order: displayOrderValue(race) }}
                    >
                      <RaceCard race={race} />
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
          {!courseGroups.length ? (
            <div className="race-grid-empty">表示できる公開レースはありません。</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
