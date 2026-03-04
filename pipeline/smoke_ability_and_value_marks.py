import csv
import tempfile
from pathlib import Path

from web_data import view_data
from web_note import build_mark_note_text


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path):
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
    run_id = "20990102_000000"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        scope_dir = base / "local"
        scope_dir.mkdir(parents=True, exist_ok=True)

        pred_rows = [
            {"HorseName": "H1", "horse_no": "1", "Top3Prob_model": "0.65", "rank_score": "2.8"},
            {"HorseName": "H2", "horse_no": "2", "Top3Prob_model": "0.58", "rank_score": "2.3"},
            {"HorseName": "H3", "horse_no": "3", "Top3Prob_model": "0.52", "rank_score": "2.0"},
            {"HorseName": "H4", "horse_no": "4", "Top3Prob_model": "0.48", "rank_score": "1.5"},
            {"HorseName": "H5", "horse_no": "5", "Top3Prob_model": "0.16", "rank_score": "0.2"},
            {"HorseName": "H6", "horse_no": "6", "Top3Prob_model": "0.44", "rank_score": "1.2"},
            {"HorseName": "H7", "horse_no": "7", "Top3Prob_model": "0.41", "rank_score": "1.0"},
            {"HorseName": "H8", "horse_no": "8", "Top3Prob_model": "0.35", "rank_score": "0.8"},
            {"HorseName": "H9", "horse_no": "9", "Top3Prob_model": "0.22", "rank_score": "0.3"},
        ]
        odds_rows = [
            {"horse_no": "1", "name": "H1", "odds": "1.2"},
            {"horse_no": "2", "name": "H2", "odds": "2.8"},
            {"horse_no": "3", "name": "H3", "odds": "3.5"},
            {"horse_no": "4", "name": "H4", "odds": "4.6"},
            {"horse_no": "5", "name": "H5", "odds": "15.0"},
            {"horse_no": "6", "name": "H6", "odds": "6.2"},
            {"horse_no": "7", "name": "H7", "odds": "7.5"},
            {"horse_no": "8", "name": "H8", "odds": "8.9"},
            {"horse_no": "9", "name": "H9", "odds": "12.0"},
        ]
        plan_rows = [
            {
                "bet_type": "place",
                "horse_name": "H1",
                "horse_no": "1",
                "amount_yen": "1200",
                "expected_return_yen": "1450",
                "hit_prob_est": "0.36",
                "ev_ratio_est": "0.18",
                "edge": "0.12",
                "gate_status": "pass",
                "risk_share": "0.25",
            },
            {
                "bet_type": "win",
                "horse_name": "H5",
                "horse_no": "5",
                "amount_yen": "900",
                "expected_return_yen": "1800",
                "hit_prob_est": "0.11",
                "ev_ratio_est": "0.55",
                "edge": "0.48",
            },
            {
                "bet_type": "wide",
                "horse_name": "H1 / H5",
                "horse_no": "1-5",
                "amount_yen": "800",
                "expected_return_yen": "1400",
                "hit_prob_est": "0.16",
                "ev_ratio_est": "0.22",
                "edge": "0.18",
            },
        ]

        _write_csv(scope_dir / f"predictions_{run_id}.csv", pred_rows)
        _write_csv(scope_dir / f"odds_{run_id}.csv", odds_rows)
        _write_csv(scope_dir / f"bet_plan_{run_id}.csv", plan_rows)

        ability_rows, ability_cols = view_data.load_ability_marks_table(
            get_data_dir=lambda base_dir, scope_key: Path(base_dir) / str(scope_key),
            base_dir=base,
            load_csv_rows=_load_csv_rows,
            to_float=_to_float,
            scope_key="local",
            run_id=run_id,
            run_row=None,
        )
        value_rows, value_cols = view_data.load_value_picks_table(
            get_data_dir=lambda base_dir, scope_key: Path(base_dir) / str(scope_key),
            base_dir=base,
            load_csv_rows=_load_csv_rows,
            to_float=_to_float,
            scope_key="local",
            run_id=run_id,
            run_row=None,
        )

        assert ability_rows and value_rows
        assert "mark" in ability_cols and "recommended_bet_types" in ability_cols
        assert "value_mark" in value_cols

        # Case A: strong horse with low odds must stay top in ability marks.
        h1 = next((r for r in ability_rows if str(r.get("horse_no", "")) == "1"), None)
        assert h1 is not None
        assert h1.get("mark") in ("◎", "○")

        # Case B: value horse (pred_rank low but high EV) must appear in value picks.
        h5 = next((r for r in value_rows if str(r.get("horse_no", "")) == "5"), None)
        assert h5 is not None
        assert h5.get("value_mark") in ("★", "☆")

        pred_csv = (
            "HorseName,Top3Prob_model,Top3Prob_lgbm,Top3Prob_lr,confidence_score,risk_score\n"
            "H1,0.65,0.62,0.60,0.78,0.70\n"
            "H2,0.58,0.55,0.53,0.78,0.70\n"
            "H5,0.16,0.12,0.10,0.78,0.70\n"
        )
        note_text = build_mark_note_text(ability_rows, value_rows, predictions_filename="", predictions_csv_text=pred_csv)
        assert "【AI能力印（◎○▲△☆）】" in note_text
        assert "【EV狙い馬（★）】" in note_text
        assert "◎ 1番 H1" in note_text or "○ 1番 H1" in note_text
        assert "★ 5番 H5" in note_text or "☆ 5番 H5" in note_text

        print("OK: ability/value marks smoke passed")


if __name__ == "__main__":
    main()
