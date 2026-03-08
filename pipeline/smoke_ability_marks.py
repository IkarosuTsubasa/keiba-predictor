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
    run_id = "20990101_000001"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        scope_dir = base / "central_turf"
        scope_dir.mkdir(parents=True, exist_ok=True)

        pred_rows = [
            {"HorseName": "Alpha", "horse_no": "9", "Top3Prob_model": "0.88"},
            {"HorseName": "Beta", "horse_no": "3", "Top3Prob_model": "0.75"},
            {"HorseName": "Gamma", "horse_no": "1", "Top3Prob_model": "0.62"},
            {"HorseName": "Delta", "horse_no": "7", "Top3Prob_model": "0.51"},
            {"HorseName": "Epsilon", "horse_no": "4", "Top3Prob_model": "0.43"},
        ]
        odds_rows = [
            {"horse_no": "1", "name": "Alpha", "odds": "3.1"},
            {"horse_no": "2", "name": "Beta", "odds": "4.2"},
            {"horse_no": "3", "name": "Gamma", "odds": "8.8"},
            {"horse_no": "4", "name": "Delta", "odds": "12.1"},
            {"horse_no": "5", "name": "Epsilon", "odds": "20.0"},
        ]
        plan_rows = [
            {
                "bet_type": "place",
                "horse_name": "Alpha",
                "horse_no": "1",
                "amount_yen": "1200",
                "expected_return_yen": "1600",
                "hit_prob_est": "0.30",
                "ev_ratio_est": "0.15",
                "gate_status": "pass",
            }
        ]

        _write_csv(scope_dir / f"predictions_{run_id}.csv", pred_rows)
        _write_csv(scope_dir / f"odds_{run_id}.csv", odds_rows)
        _write_csv(scope_dir / f"bet_plan_{run_id}.csv", plan_rows)

        ability_rows, ability_cols = view_data.load_ability_marks_table(
            get_data_dir=lambda base_dir, scope_key: Path(base_dir) / str(scope_key),
            base_dir=base,
            load_csv_rows=_load_csv_rows,
            to_float=_to_float,
            scope_key="central_turf",
            run_id=run_id,
            run_row=None,
        )
        assert ability_rows, "ability marks should not be empty"
        assert "horse_no" in ability_cols, "ability marks should include horse_no"
        top = ability_rows[0]
        assert top.get("horse_name") == "Alpha", "top horse should remain Alpha"
        assert top.get("horse_no") == "1", "ability marks should prefer current odds horse_no"

        mark_rows, _ = view_data.load_mark_recommendation_table(
            get_data_dir=lambda base_dir, scope_key: Path(base_dir) / str(scope_key),
            base_dir=base,
            load_csv_rows=_load_csv_rows,
            to_float=_to_float,
            scope_key="central_turf",
            run_id=run_id,
            run_row=None,
        )
        assert mark_rows, "integrated marks should not be empty"
        alpha = next((row for row in mark_rows if row.get("horse_name") == "Alpha"), None)
        assert alpha is not None, "Alpha should appear in integrated marks"
        assert alpha.get("horse_no") == "1", "integrated marks should prefer current odds horse_no"

        print("smoke_ability_marks: OK")


if __name__ == "__main__":
    main()
