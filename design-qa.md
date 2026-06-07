# Design QA

Target: Product Design concept 3, AI investment dashboard.

Reference checked:
- `C:\Users\owner\.codex\generated_images\019ea0c3-00b8-76a2-b64a-9a7c5b444b0a\ig_0fcbb65bbe012af6016a2517dea5e48191a2ceb253ca60f11f.png`

Implementation checked:
- `http://127.0.0.1:8000/keiba`
- `http://127.0.0.1:8000/keiba/history`
- `http://127.0.0.1:8000/keiba/race/202605021211`
- `http://127.0.0.1:8000/keiba/reports`
- `http://127.0.0.1:8000/keiba/guide`

Findings:
- P0: none.
- P1: none.
- P2: none.
- P3: the current production data only has 3 public races, so the table is shorter than the 10-row reference mock. The layout structure now matches the reference direction.

Checks completed:
- Navigation uses a dashboard-style dark top rail with active tab state.
- Home screen uses the reference composition: left date/filter panel, central race table, right validation/risk panel, and bottom focused race memo.
- Public race rows now render as a true 7-column grid: race, decision, confidence, main horse, top marks, result, details.
- Result content is inside the result column; the first visible row's 7 cells share the same y-position in the browser check.
- Racecourse filtering is separated from status tabs into a compact `競馬場` selector, preventing many racecourse names from crowding the tab row.
- Filtered race context is synchronized with the right-side AI memo and the bottom focused race memo.
- The injected share button is hidden inside the dashboard table so it does not disturb the row layout.
- Race detail, history, reports, and guide pages now use the same dashboard surfaces, panel borders, KPI cards, table styling, and compact title scale as the home page.
- Race detail now prioritizes `予測印` and `印一覧`; the buy-ticket candidate section is removed from the public detail view.
- Race detail now uses a simple content flow: `予測印`, `上位馬メモ`, then `レース結果`; the old conclusion card and long prediction memo block are removed from the public detail view.
- Public home/detail text no longer surfaces `買い目`, `購入候補`, `候補なし`, or `買い判断` in the verified browser state.
- Race detail prediction memo now normalizes raw model debug wording into Japanese public-copy wording before rendering.
- Race detail routes correctly keep the top navigation state on `公開レース`.
- English UI labels in the reports page and date filter submit button were replaced with Japanese labels.
- Production build: passed.

Final result: passed.
