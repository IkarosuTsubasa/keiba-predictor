import React, {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import RaceCard from "./RaceCard";

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

export default function RaceGrid({ races, appShell = false, onVisibleRacesChange = null }) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [locationFilter, setLocationFilter] = useState("all");
  const [viewportPhase, setViewportPhase] = useState("idle");
  const [indicator, setIndicator] = useState({ left: 0, width: 0, opacity: 0 });
  const tabsRef = useRef(null);
  const buttonRefs = useRef(new Map());
  const itemRefs = useRef(new Map());
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

  return (
    <div className="race-grid-section">
      {!appShell ? (
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
        <div className="race-grid">
          {filtered.map((race) => {
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
      </div>
    </div>
  );
}
