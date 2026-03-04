import csv
import tempfile
from pathlib import Path

from web_data import view_data


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path: Path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    run_id = "20990101_000000"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        scope_dir = base / "local"
        scope_dir.mkdir(parents=True, exist_ok=True)

        pred_rows = []
        for i in range(1, 21):
            pred_rows.append(
                {
                    "HorseName": f"H{i}",
                    "horse_no": str(i),
                    "Top3Prob_model": f"{1.5 - (i * 0.05):.4f}",
                }
            )
        odds_rows = []
        for i in range(1, 22):
            odds_rows.append(
                {
                    "horse_no": str(i),
                    "name": f"H{i}",
                    "odds": f"{2.0 + i * 0.9:.1f}",
                }
            )
        plan_rows = [
            {
                "bet_type": "place",
                "horse_name": "H1",
                "horse_no": "1",
                "amount_yen": "1200",
                "expected_return_yen": "1500",
                "hit_prob_est": "0.35",
                "ev_ratio_est": "0.20",
                "gate_status": "pass",
            },
            {
                "bet_type": "wide",
                "horse_name": "H1 / H4",
                "horse_no": "1-4",
                "amount_yen": "1400",
                "expected_return_yen": "2000",
                "hit_prob_est": "0.25",
                "ev_ratio_est": "0.28",
            },
            {
                "bet_type": "wide",
                "horse_name": "H21 / H4",
                "horse_no": "21-4",
                "amount_yen": "5000",
                "expected_return_yen": "9000",
                "hit_prob_est": "0.20",
                "ev_ratio_est": "0.50",
            },
        ]

        _write_csv(scope_dir / f"predictions_{run_id}.csv", pred_rows)
        _write_csv(scope_dir / f"odds_{run_id}.csv", odds_rows)
        _write_csv(scope_dir / f"bet_plan_{run_id}.csv", plan_rows)

        rows, cols = view_data.load_mark_recommendation_table(
            get_data_dir=lambda base_dir, scope_key: Path(base_dir) / str(scope_key),
            base_dir=base,
            load_csv_rows=_load_csv_rows,
            to_float=_to_float,
            scope_key="local",
            run_id=run_id,
            run_row=None,
        )

        assert rows, "mark rows should not be empty"
        assert len(rows) <= 5, "top rows must be <= 5"
        assert "mark" in cols and "bet_types" in cols
        assert all(int(r.get("candidate_count", 0) or 0) <= 16 for r in rows), "candidate_count should be <= 16"
        assert all("race_type" in r and "confidence" in r and "gap_1_2" in r for r in rows), "meta fields missing"

        top = rows[0]
        assert top.get("horse_name") != "H21", "bet-only horse should not become ◎"
        h1 = next((r for r in rows if r.get("horse_name") == "H1"), None)
        if h1:
            assert h1.get("bet_types") == "place,wide", "bet_types order should be place,wide"

        print("OK: integrated marks smoke passed")
        print(f"race_type={rows[0].get('race_type')} confidence={rows[0].get('confidence')} gap={rows[0].get('gap_1_2')}")
        for r in rows:
            print(
                f"{r.get('mark')} {r.get('horse_no')} {r.get('horse_name')} "
                f"bet_types={r.get('bet_types')} combined={r.get('combined_score')}"
            )


if __name__ == "__main__":
    main()
