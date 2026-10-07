import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from ground_metrics import METRICS, make_ground_cost


def write_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "MNIST")
    parser.add_argument("--output", type=Path, default=Path("metric_results.json"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--cost-scale", choices=["raw", "median"], default="raw")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--folds", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--lambda-numerators", nargs="+", type=float)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Import here so ground_metrics and plotting do not require LIBSVM.
    import sinkhorn_svm_complete as base
    sizes = args.sizes or ([3000, 5000, 12000, 17000, 25000] if args.full else [400])
    repeats = args.repeats if args.repeats is not None else (6 if args.full else 1)
    folds_to_run = args.folds if args.folds is not None else (4 if args.full else 1)
    numerators = args.lambda_numerators or ([5, 7, 9, 11] if args.full else [9])
    if min(sizes) < 1 or repeats < 1 or args.max_iter < 1 or min(numerators) <= 0:
        parser.error("Sizes, repeat/iteration counts, and lambda numerators must be positive.")
    if len(set(sizes)) != len(sizes) or len(set(args.metrics)) != len(args.metrics):
        parser.error("Do not repeat a size or metric in the arguments.")
    if args.output.exists() and not args.resume:
        parser.error("Output already exists: use --resume or choose another --output filename.")

    base.DATA_DIR = args.data_dir
    base.MAX_ITER = args.max_iter
    base.LAMBDA_NUMERATORS = numerators
    images, labels = base.load_mnist(base.find_idx("train-images.idx3-ubyte"),
                                     base.find_idx("train-labels.idx1-ubyte"))
    if max(sizes) > len(images):
        parser.error("Requested subset is larger than the available dataset.")
    selected = np.random.default_rng(args.seed).permutation(len(images))[:max(sizes)]
    X = base.make_histograms(images[selected])
    y = labels[selected]
    # This identifies the selected data and base implementation when resuming.
    fingerprint = hashlib.sha256(images[selected].tobytes() + y.tobytes()).hexdigest()
    code_fingerprint = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    config = dict(N_values=sizes, metrics=args.metrics, cost_scale=args.cost_scale,
                  side=base.SIDE, resize="BICUBIC", seed=args.seed, max_iter=args.max_iter,
                  outer_repeats=repeats, outer_folds_to_run=folds_to_run,
                  lambda_numerators=numerators, t_rules=base.T_RULES,
                  c_values=base.C_VALUES, inner_repeats=base.INNER_REPEATS,
                  data_sha256=fingerprint, base_code_sha256=code_fingerprint,
                  runtime_scope="distance calculations + tuning + refit + prediction; excludes preprocessing and saving")
    payload = dict(config=config, environment=dict(python=platform.python_version(),
                   numpy=np.__version__, platform=platform.platform()), runs=[])
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload["config"] != config:
            parser.error("Saved configuration/data/code differs. Choose a new output file.")
    completed = {(r["metric"], r["N"], r["repeat"], r["fold"]) for r in payload["runs"]}
    if len(completed) != len(payload["runs"]):
        parser.error("Saved results contain duplicate run keys.")
    costs = {name: make_ground_cost(name, base.SIDE, args.cost_scale) for name in args.metrics}

    for N in sizes:
        for repeat in range(repeats):
            # Computed once per repeat, reused unchanged across ALL metrics.
            folds = base.stratified_folds(y[:N], 4, args.seed + repeat)
            for fold in range(folds_to_run):
                train_indices = folds[fold]
                test_indices = np.concatenate([folds[k] for k in range(4) if k != fold])
                cv_seed = args.seed + 100 * repeat + 10 * fold
                # Vary execution order to reduce systematic timing order effects.
                order = np.random.default_rng(cv_seed).permutation(args.metrics).tolist()
                for metric in order:
                    key = (metric, N, repeat + 1, fold + 1)
                    if key in completed:
                        print(f"Already saved: {key}", flush=True)
                        continue
                    print(f"{metric}: N={N}, repeat={repeat + 1}, fold={fold + 1}", flush=True)
                    started = perf_counter()
                    result, prediction = base.evaluate_split(
                        X[train_indices], y[train_indices], X[test_indices], y[test_indices],
                        costs[metric], cv_seed
                    )
                    result.update(metric=metric, N=N, repeat=repeat + 1, fold=fold + 1,
                                  elapsed_seconds=perf_counter() - started,
                                  cost_median=float(np.median(costs[metric])),
                                  execution_order=order,
                                  original_train_indices=selected[train_indices].tolist(),
                                  original_test_indices=selected[test_indices].tolist(),
                                  true_labels=y[test_indices].tolist(), predictions=prediction.tolist())
                    payload["runs"].append(result)
                    write_results(args.output, payload)
                    completed.add(key)
                    print(f"  Saved: error={100 * result['test_error']:.2f}%, "
                          f"time={result['elapsed_seconds']:.1f}s", flush=True)
    print(f"Results: {args.output.resolve()}")
    print(f'Plot with: python plot_metric_results.py "{args.output}"')


if __name__ == "__main__":
    main()
