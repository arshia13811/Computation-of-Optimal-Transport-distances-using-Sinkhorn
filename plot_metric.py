
python plot_metric_results.py metric_results.json --output-dir figures
python plot_metric_results.py sinkhorn_results.json --output-dir baseline_figures

Supports the earlier standalone script's Euclidean-only JSON too.
Outputs PNG and PDF figures, plus machine-readable plot_summary.json.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ground_metrics import METRICS, make_ground_cost

COLORS = {"euclidean": "#275DAD", "manhattan": "#D47A1F", "chebyshev": "#168575"}


def describe(values):
    return dict(n=len(values), mean=float(np.mean(values)),
                sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)


def save(fig, output_dir, stem):
    fig.savefig(output_dir / (stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_series(groups, metrics, key, ylabel, title, output_dir, stem, multiplier=1):
    fig, ax = plt.subplots(figsize=(8.2, 5.0), layout="constrained")
    any_values = False
    for metric in metrics:
        points = []
        for (name, N), rows in sorted(groups.items()):
            if name == metric:
                values = [multiplier * float(r[key]) for r in rows if key in r]
                if values:
                    points.append((N, describe(values)))
        if not points:
            continue
        any_values = True
        ax.plot([p[0] for p in points], [p[1]["mean"] for p in points],
                "o-", color=COLORS[metric], label=metric.title())
        for N, stat in points:
            if stat["sd"] is not None:
                ax.errorbar(N, stat["mean"], yerr=stat["sd"], color=COLORS[metric], capsize=4)
    if not any_values:
        plt.close(fig)
        return False
    sizes = sorted({N for (_, N) in groups})
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
    ax.set(xlabel="Total subset size N (approximately N/4 used for training)", ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.supxlabel("Bars: sample SD across outer runs, not confidence intervals. No bar for a single run.", fontsize=9)
    save(fig, output_dir, stem)
    return True


def plot_geometry(metrics, side, output_dir):
    # Common median normalization and common color scale make shapes comparable.
    source = (side // 2) * side + side // 2
    maps = [make_ground_cost(m, side, "median")[source].reshape(side, side) for m in metrics]
    vmax = max(float(a.max()) for a in maps)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2),
                             squeeze=False, layout="constrained")
    for ax, metric, values in zip(axes[0], metrics, maps):
        im = ax.imshow(values, vmin=0, vmax=vmax, cmap="viridis", origin="upper")
        ax.scatter([side // 2], [side // 2], marker="x", color="white", s=50)
        ax.set(title=metric.title(), xlabel="Column", ylabel="Row")
    fig.colorbar(im, ax=list(axes[0]), shrink=.8, label="Pixel cost / median of full M")
    fig.suptitle("Cost of moving mass from the marked pixel")
    save(fig, output_dir, "01_ground_geometries")


def paired_differences(runs, metrics, output_dir):
    baseline = {(r["N"], r["repeat"], r["fold"]): r for r in runs if r["metric"] == "euclidean"}
    groups = defaultdict(list)
    for row in runs:
        if row["metric"] == "euclidean":
            continue
        ref = baseline.get((row["N"], row["repeat"], row["fold"]))
        if ref is None:
            continue  # Incomplete files may not yet have the matching baseline.
        # Same fold number alone is not sufficient evidence of pairing.
        for key in ("original_train_indices", "original_test_indices", "true_labels"):
            if key not in row or key not in ref or row[key] != ref[key]:
                raise ValueError(f"Cannot pair results: mismatched/missing {key}.")
        groups[(row["metric"], row["N"])].append(
            100 * (float(row["test_error"]) - float(ref["test_error"])))
    if not groups:
        print("No paired alternatives yet: skipping comparison-to-Euclidean plot.")
        return []
    fig, ax = plt.subplots(figsize=(8.2, 5), layout="constrained")
    summary = []
    for metric in metrics:
        points = [(N, describe(values)) for (name, N), values in sorted(groups.items()) if name == metric]
        if not points:
            continue
        ax.plot([p[0] for p in points], [p[1]["mean"] for p in points], "o-",
                color=COLORS[metric], label=metric.title())
        for N, stat in points:
            if stat["sd"] is not None:
                ax.errorbar(N, stat["mean"], yerr=stat["sd"], color=COLORS[metric], capsize=4)
            summary.append(dict(metric=metric, N=N, **stat))
    ax.axhline(0, color="#555555", lw=1, linestyle="--")
    ax.set(xlabel="Total subset size N", ylabel="Alternative error − Euclidean error (percentage points)",
           title="Paired change in test error: below zero is better")
    sizes = sorted({N for (_, N) in groups})
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.supxlabel("Matched train/test splits only. Bars: sample SD of paired differences, not confidence intervals.", fontsize=9)
    save(fig, output_dir, "03_paired_error_change")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        parser.error("No completed runs are available in this file.")
    groups = defaultdict(list)
    seen = set()
    for row in runs:
        row.setdefault("metric", "euclidean")  # earlier standalone format
        if row["metric"] not in METRICS:
            parser.error(f"Unsupported metric: {row['metric']}")
        key = (row["metric"], row["N"], row["repeat"], row["fold"])
        if key in seen:
            parser.error(f"Duplicate run: {key}")
        seen.add(key)
        error = float(row["test_error"])
        if not np.isfinite(error) or not 0 <= error <= 1:
            parser.error("test_error must be a fraction between 0 and 1.")
        if "predictions" in row and "true_labels" in row:
            prediction, truth = np.asarray(row["predictions"]), np.asarray(row["true_labels"])
            if len(truth) == 0 or prediction.shape != truth.shape or not np.isclose(np.mean(prediction != truth), error):
                parser.error("Saved test error disagrees with the predictions/labels.")
        groups[(row["metric"], row["N"])].append(row)
    metrics = [m for m in METRICS if any(r["metric"] == m for r in runs)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    plot_geometry(metrics, data.get("config", {}).get("side", 20), args.output_dir)
    plot_series(groups, metrics, "test_error", "Test error (%)", "Classification error by pixel geometry",
                args.output_dir, "02_test_error", multiplier=100)
    paired = paired_differences(runs, metrics, args.output_dir)
    if not plot_series(groups, metrics, "elapsed_seconds", "Seconds per outer run",
                       "Runtime: distances + tuning + refit + prediction", args.output_dir, "04_runtime"):
        print("No recorded runtime: skipping runtime plot.")
    plot_series(groups, metrics, "test_marginal_error", "Maximum test-plan L1 marginal error",
                "Remaining constraint error after the configured iterations", args.output_dir, "05_marginal_error")
    summary = []
    expected = data.get("config", {}).get("outer_repeats", 1) * data.get("config", {}).get("outer_folds_to_run", 1)
    for (metric, N), rows in sorted(groups.items()):
        stats = dict(metric=metric, N=N, completed_runs=len(rows), expected_runs=expected,
                     test_error_percent=describe([100 * r["test_error"] for r in rows]))
        for key in ("elapsed_seconds", "test_marginal_error"):
            values = [float(r[key]) for r in rows if key in r]
            if values:
                stats[key] = describe(values)
        summary.append(stats)
        print(f"{metric:10s} N={N}: {stats['test_error_percent']['mean']:.2f}% error, "
              f"{len(rows)}/{expected} completed runs")
    (args.output_dir / "plot_summary.json").write_text(
        json.dumps(dict(source=str(args.results.resolve()), config=data.get("config", {}),
                        groups=summary, paired_change_percentage_points=paired,
                        note="SD describes variability across overlapping outer splits; it is not a confidence interval."), indent=2),
        encoding="utf-8")
    print(f"Figures saved as PNG and PDF in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
