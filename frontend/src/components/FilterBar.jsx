import React, { useEffect, useMemo, useState } from "react";

function pad(value) {
  return String(value).padStart(2, "0");
}

function normalizeDateText(value) {
  const text = String(value || "").trim();
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : "";
}

function shiftDate(dateText, offsetDays) {
  const base = normalizeDateText(dateText);
  if (!base) return "";
  const [year, month, day] = base.split("-").map(Number);
  const next = new Date(year, month - 1, day);
  next.setDate(next.getDate() + offsetDays);
  return `${next.getFullYear()}-${pad(next.getMonth() + 1)}-${pad(next.getDate())}`;
}

export default function FilterBar({ data, search, onApply, className = "" }) {
  const params = useMemo(() => new URLSearchParams(search), [search]);
  const initialDate = normalizeDateText(params.get("date") || data?.target_date || "");
  const [date, setDate] = useState(initialDate);
  const formClassName = ["app-filter-bar", className].filter(Boolean).join(" ");

  useEffect(() => {
    setDate(initialDate);
  }, [initialDate]);

  const submitDate = (nextDate) => {
    const normalized = normalizeDateText(nextDate);
    const next = new URLSearchParams();
    if (normalized) {
      next.set("date", normalized);
    }
    onApply(next.toString());
  };

  return (
    <form
      className={formClassName}
      onSubmit={(event) => {
        event.preventDefault();
        submitDate(date);
      }}
    >
      <label className="app-filter-bar__field">
        <span>対象日</span>
        <div className="app-filter-bar__date-shell">
          <input
            type="date"
            name="date"
            className="app-filter-bar__date-input"
            value={date}
            onChange={(event) => setDate(event.target.value)}
          />
        </div>
      </label>

      <div className="app-filter-bar__actions">
        <button
          type="button"
          className="app-filter-bar__step"
          onClick={() => {
            const nextDate = shiftDate(date || initialDate, -1);
            if (!nextDate) return;
            setDate(nextDate);
            submitDate(nextDate);
          }}
        >
          前日
        </button>

        <button
          type="button"
          className="app-filter-bar__step"
          onClick={() => {
            const nextDate = shiftDate(date || initialDate, 1);
            if (!nextDate) return;
            setDate(nextDate);
            submitDate(nextDate);
          }}
        >
          翌日
        </button>

        <button type="submit" className="app-filter-bar__submit">
          Go
        </button>
      </div>
    </form>
  );
}
