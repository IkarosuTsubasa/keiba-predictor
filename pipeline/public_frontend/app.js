const app = document.getElementById("app");
const APP_BASE_PATH = "/keiba";
const PUBLIC_BOARD_API_PATH = `${APP_BASE_PATH}/api/public/board`;

function formatYen(value) {
  const amount = Number(value || 0);
  const sign = amount < 0 ? "-" : "";
  return `${sign}¥${Math.abs(amount).toLocaleString("ja-JP")}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function currentParams() {
  return new URLSearchParams(window.location.search);
}

function buildQueryFromForm(form) {
  const params = new URLSearchParams();
  const formData = new FormData(form);
  for (const [key, value] of formData.entries()) {
    const text = String(value ?? "").trim();
    if (text) {
      params.set(key, text);
    }
  }
  return params;
}

function sortSummaryCards(cards) {
  return [...(cards || [])].sort((a, b) => {
    const profitDiff = Number(b.profit_yen || 0) - Number(a.profit_yen || 0);
    if (profitDiff !== 0) return profitDiff;
    return Number(b.races || 0) - Number(a.races || 0);
  });
}

function pickLeaders(cards) {
  return sortSummaryCards(cards).slice(0, 3);
}

function renderLoading() {
  app.innerHTML = `
    <main class="state-shell">
      <section class="state-card">
        <div class="state-badge">LOADING</div>
        <h1>公開ボードを読み込み中です</h1>
        <p>最新の公開予想を整理しています。数秒で表示されます。</p>
      </section>
    </main>
  `;
}

function renderError(message) {
  app.innerHTML = `
    <main class="state-shell">
      <section class="state-card state-card--error">
        <div class="state-badge">ERROR</div>
        <h1>公開ページを読み込めませんでした</h1>
        <p>${escapeHtml(message || "予期しないエラーが発生しました。")}</p>
        <button type="button" id="retry-load">再読み込み</button>
      </section>
    </main>
  `;
  document.getElementById("retry-load")?.addEventListener("click", () => {
    loadBoard();
  });
}

function renderEmptyState() {
  return `
    <section class="empty-state">
      <div class="empty-badge">NO DATA</div>
      <h2>この日の公開データはまだありません</h2>
      <p>予想が公開されると、ここに各モデルの買い目、戦略、印、結果がまとめて表示されます。</p>
    </section>
  `;
}

function renderLeaderCards(cards) {
  if (!cards.length) {
    return `<p class="section-empty">この日の上位モデルはまだ集計できません。</p>`;
  }
  return cards
    .map((item, index) => {
      const crown = index === 0 ? "TOP" : `NO.${index + 1}`;
      return `
        <article class="leader-card leader-card--${index === 0 ? "primary" : "sub"}">
          <div class="leader-head">
            <span class="leader-rank">${crown}</span>
            <strong>${escapeHtml(item.label)}</strong>
          </div>
          <div class="leader-profit">${escapeHtml(formatYen(item.profit_yen || 0))}</div>
          <div class="leader-metrics">
            <span>${escapeHtml(item.races)}レース</span>
            <span>的中 ${escapeHtml(item.hit_races)}</span>
            <span>投資 ${escapeHtml(formatYen(item.stake_yen || 0))}</span>
            <span>払戻 ${escapeHtml(formatYen(item.payout_yen || 0))}</span>
            <span>ROI ${escapeHtml(item.roi_text || "-")}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSummaryCards(cards) {
  if (!cards.length) {
    return `<p class="section-empty">この日のモデル集計はまだありません。</p>`;
  }
  return cards
    .map(
      (item) => `
        <article class="summary-card">
          <div class="summary-card-head">
            <div>
              <span class="summary-label">${escapeHtml(item.label)}</span>
              <strong>${escapeHtml(formatYen(item.profit_yen || 0))}</strong>
            </div>
            <span class="summary-races">${escapeHtml(item.races)}レース</span>
          </div>
          <div class="summary-metrics">
            <span>確定 ${escapeHtml(item.settled_races)}</span>
            <span>待機 ${escapeHtml(item.pending_races)}</span>
            <span>的中 ${escapeHtml(item.hit_races)}</span>
            <span>買い目 ${escapeHtml(item.ticket_count)}</span>
            <span>投資 ${escapeHtml(formatYen(item.stake_yen || 0))}</span>
            <span>払戻 ${escapeHtml(formatYen(item.payout_yen || 0))}</span>
            <span>ROI ${escapeHtml(item.roi_text || "-")}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderModelCard(card) {
  return `
    <article class="model-card model-card--${escapeHtml(card.status_tone || "planned")}">
      <div class="model-card-head">
        <div>
          <div class="model-engine">${escapeHtml(card.label)}</div>
          <div class="model-decision">判定: ${escapeHtml(card.decision_text || "-")}</div>
        </div>
        <span class="model-badge model-badge--${escapeHtml(card.status_tone || "planned")}">${escapeHtml(card.status_label || "-")}</span>
      </div>
      <div class="model-body">
        <section class="model-block">
          <h4>買い目</h4>
          <p>${escapeHtml(card.ticket_plan_text || "").replaceAll("\n", "<br>")}</p>
        </section>
        <section class="model-block">
          <h4>戦略</h4>
          <p>${escapeHtml(card.strategy_text || "-")}</p>
          <p class="model-subtext">${escapeHtml(card.tendency_text || "-")}</p>
        </section>
        <section class="model-block">
          <h4>印</h4>
          <p>${escapeHtml(card.marks_text || "-")}</p>
        </section>
        <section class="model-block">
          <h4>印の結果</h4>
          <p>${escapeHtml(card.result_triplet_text || "-")}</p>
        </section>
      </div>
      <div class="model-kpis">
        <span>${escapeHtml(card.ticket_count)}点</span>
        <span>投資 ${escapeHtml(formatYen(card.stake_yen || 0))}</span>
        <span>払戻 ${escapeHtml(formatYen(card.payout_yen || 0))}</span>
        <span>収支 ${escapeHtml(formatYen(card.profit_yen || 0))}</span>
        <span>的中 ${escapeHtml(card.hit_count)}</span>
        <span>ROI ${escapeHtml(card.roi_text || "-")}</span>
      </div>
    </article>
  `;
}

function renderRaceBoards(races) {
  return (races || [])
    .map((race, index) => {
      const cards = (race.cards || []).map(renderModelCard).join("");
      return `
        <details class="race-board" ${index < 2 ? "open" : ""}>
          <summary class="race-summary">
            <div class="race-copy">
              <span class="race-scope">${escapeHtml(race.scope_label || "-")}</span>
              <h2>${escapeHtml(race.race_title || "-")}</h2>
              <p>${escapeHtml(race.date_label || "-")}</p>
            </div>
            <div class="race-meta">
              <span>${escapeHtml(race.actual_text || "結果未確定")}</span>
              <span>${escapeHtml((race.cards || []).length)}モデル</span>
              <span>Run ${escapeHtml(race.run_id || "-")}</span>
            </div>
          </summary>
          <div class="race-grid">${cards}</div>
        </details>
      `;
    })
    .join("");
}

function renderBoard(data) {
  const totals = data.totals || {};
  const summaryCards = sortSummaryCards(data.summary_cards || []);
  const leaders = pickLeaders(summaryCards);
  const races = data.races || [];
  const scopeOptions = (data.scope_options || [])
    .map((item) => {
      const selected = item.value === (data.scope_key || "") ? " selected" : "";
      return `<option value="${escapeHtml(item.value)}"${selected}>${escapeHtml(item.label)}</option>`;
    })
    .join("");

  const fallbackNotice = data.fallback_notice
    ? `<section class="notice">${escapeHtml(data.fallback_notice)}</section>`
    : "";

  app.innerHTML = `
    <main class="app-shell">
      <section class="hero">
        <div class="hero-head">
          <div class="hero-copy">
            <div class="eyebrow">PUBLIC RACING BOARD</div>
            <h1>公開されたAI予想を、読みやすく。</h1>
            <p>今日公開されている各LLMの買い目、戦略、印、収支を1ページに集約。初見でも流れを追いやすいように、先に全体像、その次に上位モデル、最後に各レースの詳細を見せています。</p>
          </div>
          <div class="hero-feature">
            <span class="eyebrow">TODAY SNAPSHOT</span>
            <strong>${escapeHtml(formatYen(totals.profit_yen || 0))}</strong>
            <p>総投資 ${escapeHtml(formatYen(totals.stake_yen || 0))} / 総払戻 ${escapeHtml(formatYen(totals.payout_yen || 0))}</p>
            <p>ROI ${escapeHtml(totals.roi_text || "-")} ・ 確定 ${escapeHtml(totals.settled_count || 0)} レース ・ 結果待ち ${escapeHtml(totals.pending_count || 0)} レース</p>
          </div>
        </div>
        <div class="hero-strip">
          <article class="hero-stat">
            <span>表示日</span>
            <strong>${escapeHtml(data.target_date_label || "-")}</strong>
          </article>
          <article class="hero-stat">
            <span>公開レース数</span>
            <strong>${escapeHtml(totals.race_count || 0)}</strong>
          </article>
          <article class="hero-stat">
            <span>稼働モデル</span>
            <strong>${escapeHtml(totals.engine_count || 0)}</strong>
          </article>
          <article class="hero-stat">
            <span>総収支</span>
            <strong>${escapeHtml(formatYen(totals.profit_yen || 0))}</strong>
          </article>
        </div>
        <form class="filters" id="board-filters">
          <label class="filter-field">日付
            <input type="date" name="date" value="${escapeHtml(data.target_date || "")}">
          </label>
          <label class="filter-field">範囲
            <select name="scope_key">${scopeOptions}</select>
          </label>
          <button type="submit">表示を更新</button>
        </form>
      </section>
      ${fallbackNotice}
      <section class="section leaders">
        <div class="section-head">
          <div>
            <div class="eyebrow">TODAY LEADERS</div>
            <h2>今日の上位モデル</h2>
          </div>
          <p>その日の公開結果をもとに、まず目に入るべきモデルを前に出しています。</p>
        </div>
        <div class="leaders-grid">${renderLeaderCards(leaders)}</div>
      </section>
      <section class="section">
        <div class="section-head">
          <div>
            <div class="eyebrow">MODEL SUMMARY</div>
            <h2>モデル別サマリー</h2>
          </div>
          <p>全レースをモデル単位で集計した一覧です。利益、的中、投資、払戻を比較できます。</p>
        </div>
        <div class="summary-grid">${renderSummaryCards(summaryCards)}</div>
      </section>
      ${
        races.length
          ? `
            <section class="section section--races">
              <div class="section-head">
                <div>
                  <div class="eyebrow">RACE DETAILS</div>
                  <h2>レース別の公開内容</h2>
                </div>
                <p>各レースは折りたたみで整理しています。気になるレースだけ開いて詳細を確認できます。</p>
              </div>
              <div class="races-stack">${renderRaceBoards(races)}</div>
            </section>
          `
          : renderEmptyState()
      }
    </main>
  `;

  document.getElementById("board-filters")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const params = buildQueryFromForm(event.currentTarget);
    const nextUrl = params.toString() ? `${APP_BASE_PATH}?${params.toString()}` : APP_BASE_PATH;
    window.history.pushState({}, "", nextUrl);
    loadBoard();
  });
}

async function loadBoard() {
  renderLoading();
  const params = currentParams();
  const query = new URLSearchParams();
  const date = params.get("date");
  const scopeKey = params.get("scope_key");
  if (date) {
    query.set("date", date);
  }
  if (scopeKey) {
    query.set("scope_key", scopeKey);
  }
  const url = query.toString() ? `${PUBLIC_BOARD_API_PATH}?${query.toString()}` : PUBLIC_BOARD_API_PATH;
  try {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    renderBoard(payload);
  } catch (error) {
    renderError(error?.message || "データの取得に失敗しました。");
  }
}

window.addEventListener("popstate", () => {
  loadBoard();
});

loadBoard();
