import argparse
import json
import sys
from pathlib import Path

from local_env import load_local_env


BASE_DIR = Path(__file__).resolve().parent
load_local_env(BASE_DIR, override=False)


def _build_parser():
    parser = argparse.ArgumentParser(description="Run one policy engine in an isolated subprocess")
    parser.add_argument("--scope-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-engine", required=True)
    parser.add_argument("--policy-model", default="")
    parser.add_argument("--output-json", required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    scope_key = str(args.scope_key or "").strip()
    run_id = str(args.run_id or "").strip()
    policy_engine = str(args.policy_engine or "").strip()
    policy_model = str(args.policy_model or "").strip()
    output_path = Path(str(args.output_json or "").strip())

    if not scope_key or not run_id or not policy_engine or not str(output_path):
        print("missing required arguments")
        return 2

    import web_app

    run_row = web_app.resolve_run(run_id, scope_key)
    if run_row is None:
        print(f"run row not found for run_id={run_id} scope_key={scope_key}")
        return 2

    result = web_app.execute_policy_buy(
        scope_key,
        dict(run_row),
        run_id,
        policy_engine=policy_engine,
        policy_model=policy_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "scope_key": scope_key,
                "run_id": run_id,
                "policy_engine": policy_engine,
                "output_json": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
