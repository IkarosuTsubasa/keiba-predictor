import React from "react";

export default function FilterBar({ data, search, onApply }) {
  const params = new URLSearchParams(search);

  return (
    <form
      className="app-filter-bar"
      onSubmit={(event) => {
        event.preventDefault();
        const formData = new FormData(event.currentTarget);
        const next = new URLSearchParams();
        const date = String(formData.get("date") || "").trim();
        if (date) next.set("date", date);
        onApply(next.toString());
      }}
    >
      <label className="app-filter-bar__field">
        <span>対象日</span>
        <div className="app-filter-bar__date-shell">
          <input
            type="date"
            name="date"
            className="app-filter-bar__date-input"
            defaultValue={params.get("date") || data?.target_date || ""}
          />
        </div>
      </label>

      <button type="submit" className="app-filter-bar__submit">
        更新
      </button>
    </form>
  );
}
