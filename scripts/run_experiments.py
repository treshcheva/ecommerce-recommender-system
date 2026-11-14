"""Execute existing notebook experiments; searches remain inside notebooks."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from importlib.metadata import version

import pandas as pd

from src.notebook_execution import prepare_notebook, save_executed_notebook
from src.benchmark import artifact_root, estimate_benchmark_cost, get_benchmark_config
from src.retailrocket import load_processed_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "full"], default="dev")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--verify-reproducibility", action="store_true",
                        help="Repeat and compare metrics/configurations with existing results.")
    args = parser.parse_args()
    if args.mode == "full" and not args.estimate_only:
        if args.verify_reproducibility:
            parser.error("Full execution resumes successful stages; automatic heavy replay is disabled.")
        from scripts.run_full import run_logged
        run_logged()
        return
    root = Path(__file__).resolve().parents[1]
    config = get_benchmark_config(args.mode)
    destination = artifact_root(root, config)
    metrics = destination / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    # Always preflight before starting any notebook, using train only.
    train = load_processed_events(root / "data/processed/train.csv")
    estimate = estimate_benchmark_cost(train)
    estimate.to_csv(metrics / "benchmark_cost_estimate.csv", index=False)
    print(estimate.to_string(index=False), flush=True)
    print("Memory: planning allowances, not measured RSS. Ratios are work proxies, not seconds.", flush=True)
    if args.estimate_only:
        return
    del train
    previous = {}
    comparison_files = [
        "model_validation_metrics.csv", "model_test_metrics.csv",
        "itemknn_tuning_validation.csv", "implicit_als_tuning_validation.csv",
        "model_validation_paired_intervals.csv", "model_test_paired_intervals.csv",
        "ltr_validation_metrics.csv", "ltr_test_metrics.csv",
        "ltr_catboost_tuning_validation.csv", "ltr_ablation_validation.csv",
        "ltr_ablation_paired_intervals.csv", "ltr_test_paired_intervals.csv",
        "ltr_candidate_recall.csv", "ltr_test_per_user.csv", "ltr_ablation_per_user.csv",
    ]
    if args.verify_reproducibility:
        for filename in comparison_files:
            previous[filename] = pd.read_csv(metrics / filename).drop(columns=["fit_seconds"], errors="ignore")
        previous_selection = json.loads((metrics / "selected_model_params.json").read_text())
    from nbclient import NotebookClient
    os.environ["BENCHMARK_MODE"] = args.mode
    os.environ["RESUME_EXPERIMENTS"] = "0"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    run = {
        "mode": args.mode, "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ["numpy", "pandas", "scipy", "implicit", "catboost", "matplotlib", "nbclient"]},
        "notebook_seconds": {},
    }
    # Use the invoking interpreter, rather than a potentially unrelated kernel.
    with tempfile.TemporaryDirectory(prefix="retailrocket-kernel-") as directory:
        kernel = Path(directory) / "kernels" / "retailrocket"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Retailrocket", "language": "python",
        }))
        old_path = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = directory + (os.pathsep + old_path if old_path else "")
        for name in ["02_modeling", "03_evaluation", "04_learning_to_rank"]:
            source = root / "notebooks" / (name + ".ipynb")
            target = destination / "dev" / "notebooks" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            notebook = prepare_notebook(source, mode=args.mode, resume=False)
            started = time.perf_counter()
            print("Executing", name, args.mode, flush=True)
            NotebookClient(notebook, timeout=1800, kernel_name="retailrocket",
                           resources={"metadata": {"path": str(root)}}).execute()
            save_executed_notebook(notebook, source, target)
            run["notebook_seconds"][name] = time.perf_counter() - started
            (metrics / "execution_manifest.json").write_text(json.dumps(run, indent=2))
            print("Finished", name, flush=True)

    if args.verify_reproducibility:
        for filename, expected in previous.items():
            actual = pd.read_csv(metrics / filename).drop(columns=["fit_seconds"], errors="ignore")
            pd.testing.assert_frame_equal(actual, expected, rtol=1e-7, atol=1e-9)
        assert previous_selection == json.loads((metrics / "selected_model_params.json").read_text())
        report = {"mode": args.mode, "status": "passed", "tables_compared": comparison_files,
                  "selected_parameters_equal": True, "rtol": 1e-7, "atol": 1e-9,
                  "excluded_columns": ["fit_seconds"],
                  "note": "Same data, environment, seed and search; repetition verifies reproducibility, not independent generalization."}
        (metrics / "reproducibility_check.json").write_text(json.dumps(report, indent=2))
        print("Reproducibility verified:", len(previous), "tables and selected parameters", flush=True)


if __name__ == "__main__":
    main()
