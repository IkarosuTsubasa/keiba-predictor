import shutil
import tempfile
from pathlib import Path

from gemini_portfolio import (
    build_history_context,
    extract_ledger_date,
    load_daily_profit_rows,
    reserve_run_tickets,
    settle_run_tickets,
    summarize_bankroll,
)


def assert_true(cond, message):
    if not cond:
        raise AssertionError(message)


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8-sig")


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="gemini_portfolio_smoke_", dir=str(Path.cwd())))
    try:
        base_dir = tmp_root
        ledger_date = extract_ledger_date("20260309_123456")
        reserve_run_tickets(
            base_dir,
            run_id="20260309_123456",
            scope_key="central_dirt",
            race_id="202603090101",
            ledger_date=ledger_date,
            tickets=[
                {
                    "ticket_id": "place-4",
                    "bet_type": "place",
                    "horse_no": "4",
                    "horse_name": "Alpha",
                    "amount_yen": 400,
                    "odds_used": 1.8,
                },
                {
                    "ticket_id": "wide-4-6",
                    "bet_type": "wide",
                    "horse_no": "4-6",
                    "horse_name": "Alpha / Beta",
                    "amount_yen": 300,
                    "odds_used": 3.5,
                },
            ],
        )
        summary = summarize_bankroll(base_dir, ledger_date)
        assert_true(summary["available_bankroll_yen"] == 9300, "reserve did not reduce bankroll")
        other_summary = summarize_bankroll(base_dir, ledger_date, policy_engine="siliconflow")
        assert_true(other_summary["available_bankroll_yen"] == 10000, "other provider bankroll should stay isolated")

        odds_dir = base_dir / "data" / "central_dirt" / "202603090101"
        odds_path = odds_dir / "odds.csv"
        fuku_path = odds_dir / "fuku_odds.csv"
        wide_path = odds_dir / "wide_odds.csv"
        write_text(
            odds_path,
            "horse_no,name,odds\n4,Alpha,2.2\n6,Beta,4.1\n8,Gamma,9.9\n",
        )
        write_text(
            fuku_path,
            "horse_no,odds_low,odds_high,odds_mid\n4,1.8,2.0,1.9\n6,2.3,2.5,2.4\n8,3.0,3.2,3.1\n",
        )
        write_text(
            wide_path,
            "horse_no_a,horse_no_b,odds_mid\n4,6,3.5\n4,8,5.0\n6,8,6.0\n",
        )

        settlement = settle_run_tickets(
            base_dir,
            {
                "run_id": "20260309_123456",
                "timestamp": "2026-03-09T12:34:56",
                "odds_path": str(odds_path),
                "fuku_odds_path": str(fuku_path),
                "wide_odds_path": str(wide_path),
                "quinella_odds_path": "",
            },
            ["Alpha", "Beta", "Gamma"],
        )
        assert_true(bool(settlement), "settlement missing")
        assert_true(settlement["run_profit_yen"] == 1070, "unexpected settlement profit")

        summary_after = summarize_bankroll(base_dir, ledger_date)
        assert_true(summary_after["available_bankroll_yen"] == 11070, "settlement did not restore bankroll")
        other_summary_after = summarize_bankroll(base_dir, ledger_date, policy_engine="siliconflow")
        assert_true(other_summary_after["available_bankroll_yen"] == 10000, "other provider bankroll changed unexpectedly")
        reserve_run_tickets(
            base_dir,
            run_id="20260309_223344",
            scope_key="central_dirt",
            race_id="202603090102",
            ledger_date=ledger_date,
            tickets=[
                {
                    "ticket_id": "place-2",
                    "bet_type": "place",
                    "horse_no": "2",
                    "horse_name": "Delta",
                    "amount_yen": 500,
                    "odds_used": 2.4,
                }
            ],
            policy_engine="siliconflow",
        )
        siliconflow_summary = summarize_bankroll(base_dir, ledger_date, policy_engine="siliconflow")
        assert_true(siliconflow_summary["available_bankroll_yen"] == 9500, "siliconflow reserve should use its own bankroll")
        gemini_summary_final = summarize_bankroll(base_dir, ledger_date)
        assert_true(gemini_summary_final["available_bankroll_yen"] == 11070, "gemini bankroll changed after siliconflow reserve")

        daily = load_daily_profit_rows(base_dir, days=30)
        assert_true(daily and int(daily[0]["profit_yen"]) == 1070, "daily profit summary mismatch")
        history = build_history_context(base_dir, ledger_date, lookback_days=14, recent_ticket_limit=4)
        assert_true(int(history["today"]["available_bankroll_yen"]) == 11070, "history today bankroll mismatch")
        assert_true(int(history["lookback_summary"]["profit_yen"]) == 1070, "history lookback profit mismatch")
        assert_true(bool(history["recent_tickets"]), "history recent tickets missing")
        print("smoke_gemini_portfolio: OK")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_gemini_portfolio: FAIL: {exc}")
        raise
