import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from eval_ranker_modes import evaluate_scope, save_eval_csv
from surface_scope import get_data_dir, get_predictor_config_path, normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCOPES = ["central_dirt", "central_turf", "local"]


def _parse_scopes(text: str) -> List[str]:
    scopes = []
    for token in str(text or "").split(","):
        key = normalize_scope_key(token.strip())
        if key and key not in scopes:
            scopes.append(key)
    return scopes or list(DEFAULT_SCOPES)


def _rank_key(stats: Dict[str, float]) -> Tuple[float, float, float]:
    primary = float(stats["hit_at_5_mean"]) - 0.5 * float(stats["hit_at_5_std"])
    secondary = float(stats["top3_hits_at_5_mean"])
    tertiary = float(stats["mrr_top3_mean"])
    return primary, secondary, tertiary


def _pick_best_alpha(hybrid_metrics: Dict[float, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    best_alpha = None
    best_stats = None
    for alpha, stats in hybrid_metrics.items():
        if best_alpha is None:
            best_alpha = float(alpha)
            best_stats = stats
            continue
        if _rank_key(stats) > _rank_key(best_stats):
            best_alpha = float(alpha)
            best_stats = stats
    if best_alpha is None or best_stats is None:
        raise ValueError("No hybrid metrics to rank.")
    return best_alpha, best_stats


def _render_config_patch(config_path: Path, old_text: str, new_text: str) -> str:
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=str(config_path),
        tofile=str(config_path),
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Tune hybrid ranker alpha by grouped race evaluation.")
    parser.add_argument("--scopes", default="central_dirt,central_turf,local", help="Comma-separated scopes")
    parser.add_argument("--split-count", type=int, default=5, help="Number of grouped splits")
    parser.add_argument("--seeds", default="11,22,33,44,55", help="Comma-separated random seeds")
    parser.add_argument("--alpha-grid", default="0.3:0.8:0.05", help="Grid like 0.3:0.8:0.05 or csv")
    parser.add_argument("--test-size", type=float, default=0.2, help="Grouped holdout test size")
    parser.add_argument("--apply", action="store_true", help="Apply best alpha directly to predictor config json")
    args = parser.parse_args()

    scopes = _parse_scopes(args.scopes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_chunks = []
    summary_rows = []

    for scope in scopes:
        result = evaluate_scope(
            scope_key=scope,
            split_count=args.split_count,
            seed_text=args.seeds,
            alpha_grid_text=args.alpha_grid,
            test_size=args.test_size,
        )
        eval_out = get_data_dir(BASE_DIR, scope) / f"eval_ranker_modes_{timestamp}.csv"
        save_eval_csv(result, eval_out)

        best_alpha, best_stats = _pick_best_alpha(result["hybrid"])
        baseline_stats = result["baseline"]
        ranker_stats = result["ranker"]
        cfg_path = get_predictor_config_path(BASE_DIR, scope)

        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        params = dict(cfg.get("params", {}))
        old_alpha = float(params.get("ranker_blend_alpha", 0.6))
        params["ranker_blend_alpha"] = round(float(best_alpha), 4)
        cfg["params"] = params
        new_text = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
        old_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else "{}\n"

        summary_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scope": scope,
                "splits": result["split_count"],
                "races": result["races"],
                "rows": result["rows"],
                "old_alpha": round(old_alpha, 4),
                "best_alpha": round(float(best_alpha), 4),
                "best_score_primary": round(_rank_key(best_stats)[0], 6),
                "best_hit_at_5_mean": round(best_stats["hit_at_5_mean"], 6),
                "best_hit_at_5_std": round(best_stats["hit_at_5_std"], 6),
                "best_top3_hits_at_5_mean": round(best_stats["top3_hits_at_5_mean"], 6),
                "best_top3_hits_at_5_std": round(best_stats["top3_hits_at_5_std"], 6),
                "best_mrr_top3_mean": round(best_stats["mrr_top3_mean"], 6),
                "best_mrr_top3_std": round(best_stats["mrr_top3_std"], 6),
                "baseline_hit_at_5_mean": round(baseline_stats["hit_at_5_mean"], 6),
                "baseline_top3_hits_at_5_mean": round(baseline_stats["top3_hits_at_5_mean"], 6),
                "baseline_mrr_top3_mean": round(baseline_stats["mrr_top3_mean"], 6),
                "ranker_hit_at_5_mean": round(ranker_stats["hit_at_5_mean"], 6),
                "ranker_top3_hits_at_5_mean": round(ranker_stats["top3_hits_at_5_mean"], 6),
                "ranker_mrr_top3_mean": round(ranker_stats["mrr_top3_mean"], 6),
                "eval_csv": str(eval_out),
                "config_path": str(cfg_path),
            }
        )

        print(f"\n[{scope}]")
        print(
            "baseline: "
            f"hit@5={baseline_stats['hit_at_5_mean']:.4f}+/-{baseline_stats['hit_at_5_std']:.4f}, "
            f"top3@5={baseline_stats['top3_hits_at_5_mean']:.4f}+/-{baseline_stats['top3_hits_at_5_std']:.4f}, "
            f"mrr@10={baseline_stats['mrr_top3_mean']:.4f}+/-{baseline_stats['mrr_top3_std']:.4f}"
        )
        print(
            "ranker  : "
            f"hit@5={ranker_stats['hit_at_5_mean']:.4f}+/-{ranker_stats['hit_at_5_std']:.4f}, "
            f"top3@5={ranker_stats['top3_hits_at_5_mean']:.4f}+/-{ranker_stats['top3_hits_at_5_std']:.4f}, "
            f"mrr@10={ranker_stats['mrr_top3_mean']:.4f}+/-{ranker_stats['mrr_top3_std']:.4f}"
        )
        print(
            f"best hybrid alpha={best_alpha:.2f}: "
            f"hit@5={best_stats['hit_at_5_mean']:.4f}+/-{best_stats['hit_at_5_std']:.4f}, "
            f"top3@5={best_stats['top3_hits_at_5_mean']:.4f}+/-{best_stats['top3_hits_at_5_std']:.4f}, "
            f"mrr@10={best_stats['mrr_top3_mean']:.4f}+/-{best_stats['mrr_top3_std']:.4f}"
        )

        if args.apply:
            cfg_path.write_text(new_text, encoding="utf-8")
            print(f"applied: {cfg_path} (ranker_blend_alpha={best_alpha:.2f})")
        else:
            patch_text = _render_config_patch(cfg_path, old_text, new_text)
            patch_chunks.append(patch_text)

    summary_df = pd.DataFrame(summary_rows)
    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"tune_ranker_alpha_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nsummary saved: {summary_path}")

    if not args.apply:
        patch_path = out_dir / f"tune_ranker_alpha_patch_{timestamp}.diff"
        patch_text = "\n".join(chunk for chunk in patch_chunks if chunk.strip())
        patch_path.write_text(patch_text, encoding="utf-8")
        print(f"patch saved: {patch_path}")
        if patch_text.strip():
            print("\n--- suggested patch ---")
            print(patch_text)
        else:
            print("\nNo config changes suggested.")


if __name__ == "__main__":
    main()
