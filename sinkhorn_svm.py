"""MNIST Sinkhorn + one-vs-one LIBSVM, without scikit-learn.

Install: python -m pip install numpy scipy Pillow libsvm-official==3.37.0
Run:     python sinkhorn_svm_complete.py

Default: small development run. Set FULL_EXPERIMENT=True for the larger
Sinkhorn classification experiment (not the paper's other baselines/timings).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import eigh
from libsvm.svmutil import svm_predict, svm_problem, svm_train


# 1. SETTINGS: every experiment parameter is defined before it is used.
DATA_DIR = Path(__file__).resolve().parent / "MNIST"
OUTPUT_FILE = Path(__file__).resolve().parent / "sinkhorn_results.json"
FULL_EXPERIMENT = True
SIDE = 20
MAX_ITER = 20
BATCH_SIZE = 256
SEED = 42

# N is TOTAL subset size; approximately N/4 images train, 3N/4 test.
N_VALUES = [3000, 5000, 12000, 17000, 25000] if FULL_EXPERIMENT else [400]
OUTER_REPEATS = 6 if FULL_EXPERIMENT else 1
OUTER_FOLDS_TO_RUN = 4 if FULL_EXPERIMENT else 1
LAMBDA_NUMERATORS = [5, 7, 9, 11] if FULL_EXPERIMENT else [9]
T_RULES = ["one", "q10", "q20", "q50"]
# Final NeurIPS proceedings grid. Some other versions include 100 as well.
C_VALUES = [0.01, 1.0, 10000.0]
INNER_REPEATS = 2


# 2. LOAD MATCHING IMAGE AND LABEL FILES.
def find_idx(stem):
    for name in (stem, stem.replace(".", "-")):
        for suffix in ("", ".gz"):
            path = DATA_DIR / (name + suffix)
            if path.is_file():
                return path
    raise FileNotFoundError(f"Put {stem}, or its .gz version, in {DATA_DIR}")


def read_bytes(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return f.read()


def load_mnist(image_path, label_path):
    image_bytes = read_bytes(image_path)
    label_bytes = read_bytes(label_path)
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError("An IDX header is incomplete.")
    magic, count, rows, cols = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if magic != 2051 or label_magic != 2049:
        raise ValueError("Expected image magic 2051 and label magic 2049.")
    if count != label_count or count == 0 or rows == 0 or cols == 0:
        raise ValueError("Image and label counts must match and dimensions be positive.")
    pixels = np.frombuffer(image_bytes, dtype=np.uint8, offset=16)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8).astype(int)
    if pixels.size != count * rows * cols or labels.size != count:
        raise ValueError("IDX data length does not match its header.")
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be integers from 0 to 9.")
    return pixels.reshape(count, rows, cols), labels


# 3. TURN EACH IMAGE INTO A PROBABILITY VECTOR; BUILD PIXEL COSTS.
def make_histograms(images, side=SIDE):
    normalized_histogram = []
    for image in images:
        resized = Image.fromarray(image).resize(
            (side, side), resample=Image.Resampling.BICUBIC
        )
        flat_array = np.asarray(resized, dtype=np.float64).flatten()
        pixel_sum = flat_array.sum()
        if pixel_sum <= 0:
            raise ValueError("A completely black image cannot be normalized.")
        normalized_histogram.append(flat_array / pixel_sum)

    # OUTSIDE the loop: convert the list into a 2D matrix with .T available.
    normalized_histogram = np.asarray(normalized_histogram, dtype=np.float64)
    return normalized_histogram


def make_cost_matrix(side=SIDE):
    coordinates = np.array([(x, y) for x in range(side) for y in range(side)])
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=2)).astype(np.float64)


# 4. SINKHORN: ONE SOURCE r AGAINST ALL TARGET COLUMNS IN C.
def ComputeSinkhornDistances(r, C, M, lmbdas, max_iter, K=None):
    """Return distances and the largest L1 marginal error across targets.

    r: (d,), C: (d, number_of_targets), M: (d,d).
    K can be reused because it depends only on M and lambda.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if K is None:
        K = np.exp(-lmbdas * M)
    if not np.isfinite(K).all() or np.any(K <= 0):
        raise FloatingPointError("K underflowed; this setting needs log-domain Sinkhorn.")

    # Algorithm 1 drops source bins with zero mass (black pixels).
    active = r > 0
    r = r[active]
    K = K[active, :]
    M = M[active, :]
    u = np.ones((len(r), C.shape[1])) / len(r)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for _ in range(max_iter):
            V = C / (K.T @ u)
            u = r[:, None] / (K @ V)

        # The loop ends by changing u, so update V using that FINAL u.
        V = C / (K.T @ u)
        distances = np.sum(u * ((K * M) @ V), axis=0)

        # Diagnostics only: fixed 20 iterations, not a tolerance-based stop.
        row_error = np.sum(np.abs(u * (K @ V) - r[:, None]), axis=0)
        col_error = np.sum(np.abs(V * (K.T @ u) - C), axis=0)
        error = float(max(row_error.max(), col_error.max()))
    return distances, error


def distance_matrix(X_query, X_train, M, lmbdas, max_iter, batch_size=BATCH_SIZE):
    """Rows = query images; columns = training images, always in that order."""
    for X in (X_query, X_train):
        if X.ndim != 2 or X.shape[1] != len(M) or len(X) == 0:
            raise ValueError("Expected nonempty matrices with one histogram per row.")
        if not np.isfinite(X).all() or np.any(X < 0) or not np.allclose(X.sum(1), 1):
            raise ValueError("Histograms must be finite, nonnegative, and sum to 1.")
    K = np.exp(-lmbdas * M)
    distances = np.empty((len(X_query), len(X_train)), dtype=np.float64)
    worst_error = 0.0
    for i, r in enumerate(X_query):
        for start in range(0, len(X_train), batch_size):
            stop = min(start + batch_size, len(X_train))
            C = X_train[start:stop].T
            values, error = ComputeSinkhornDistances(r, C, M, lmbdas, max_iter, K)
            distances[i, start:stop] = values
            worst_error = max(worst_error, error)
    return distances, worst_error


# 5. CONVERT IMAGE DISTANCES INTO AN SVM KERNEL AND FIX ITS EIGENVALUES.
def bandwidth(train_dist, rule):
    # Your np.asarray(train_dist).flatten() was already correct.
    dist_flat = np.asarray(train_dist, dtype=np.float64).flatten()
    quantiles = {"q10": 0.1, "q20": 0.2, "q50": 0.5}
    t = 1.0 if rule == "one" else float(np.quantile(dist_flat, quantiles[rule]))
    if t <= 0 or not np.isfinite(t):
        raise ValueError("A kernel bandwidth t must be strictly positive.")
    return t


def make_psd(train_kernel, margin=1e-8):
    # Finite Sinkhorn iterations can produce directional asymmetry.
    S = (train_kernel + train_kernel.T) / 2
    smallest = float(eigh(S, subset_by_index=[0, 0], eigvals_only=True)[0])
    shift = max(0.0, margin - smallest)
    S = S + shift * np.eye(len(S))
    return S, smallest, shift


# 6. LIBSVM WRAPPERS: THE MULTICLASS ONE-VS-ONE LOGIC IS BUILT IN.
def libsvm_rows(S):
    # LIBSVM precomputed input requires a leading 1-based sample index.
    # ADD .tolist() HERE
    return np.column_stack((np.arange(1, len(S) + 1), S)).tolist()


def fit_svm(train_kernel, y_train, C_svm):
    # ADD .tolist() TO y_train HERE
    problem = svm_problem(y_train.tolist(), libsvm_rows(train_kernel), isKernel=True)
    # -s 0: C-SVC; -t 4: precomputed kernel; -c: SVM C; -q: quiet.
    return svm_train(problem, f"-s 0 -t 4 -c {C_svm} -e 0.001 -q")


def predict_svm(model, test_kernel):
    # ADD .tolist() TO np.zeros HERE
    predicted, _, _ = svm_predict(
        np.zeros(len(test_kernel)).tolist(), libsvm_rows(test_kernel), model, "-q"
    )
    return np.asarray(predicted, dtype=int)

# 7. STRATIFIED SPLITS AND PARAMETER SELECTION, WITHOUT SCIKIT-LEARN.
def stratified_folds(labels, n_folds, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for digit in np.unique(labels):
        indices = np.flatnonzero(labels == digit)
        if len(indices) < n_folds:
            raise ValueError("Too few examples per class: increase the subset size.")
        rng.shuffle(indices)
        for k, part in enumerate(np.array_split(indices, n_folds)):
            folds[k].extend(part.tolist())
    return [rng.permutation(np.asarray(fold, dtype=int)) for fold in folds]


def CV_ct(train_dist, y_train, t_rules, c_values, seed):
    """Select t AND C together using repeated two-fold validation.

    Quantile rules are fitted separately on each inner training fold.
    Return (best_t, best_c, best_rule, best_validation_error) in ONE call.
    """
    splits = []
    for repeat in range(INNER_REPEATS):
        a, b = stratified_folds(y_train, 2, seed + repeat)
        splits.extend([(a, b), (b, a)])

    best_error = np.inf
    best_rule, best_c = None, None
    for rule in t_rules:
        # Same folds for every C and every t rule.
        errors = [[] for _ in c_values]
        for fit_indices, val_indices in splits:
            D_fit = train_dist[np.ix_(fit_indices, fit_indices)]
            D_val = train_dist[np.ix_(val_indices, fit_indices)]
            t = bandwidth(D_fit, rule)
            S_fit, _, _ = make_psd(np.exp(-D_fit / t))
            S_val = np.exp(-D_val / t)
            for j, C_svm in enumerate(c_values):
                model = fit_svm(S_fit, y_train[fit_indices], C_svm)
                prediction = predict_svm(model, S_val)
                errors[j].append(float(np.mean(prediction != y_train[val_indices])))
        for C_svm, fold_errors in zip(c_values, errors):
            mean_error = float(np.mean(fold_errors))
            if mean_error < best_error:
                best_error, best_rule, best_c = mean_error, rule, C_svm

    # Refit the selected bandwidth rule using ALL outer-training distances.
    best_t = bandwidth(train_dist, best_rule)
    return best_t, best_c, best_rule, best_error


# 8. EXPERIMENT: FUNCTIONS ABOVE ARE NOW DEFINED, SO IT IS SAFE TO CALL THEM.
def evaluate_split(X_train, y_train, X_test, y_test, M, seed):
    best = None
    median_M = float(np.median(M))
    if median_M <= 0:
        raise ValueError("The median pixel distance must be positive.")
    for numerator in LAMBDA_NUMERATORS:
        lmbdas = numerator / median_M
        print(f"  Computing training distances: lambda = {numerator}/median(M)", flush=True)
        train_dist, residual = distance_matrix(X_train, X_train, M, lmbdas, MAX_ITER)
        best_t, best_c, rule, cv_error = CV_ct(train_dist, y_train, T_RULES, C_VALUES, seed)
        if best is None or cv_error < best["cv_error"]:
            best = dict(lambda_numerator=numerator, lambda_ot=lmbdas, t=best_t,
                        C_svm=best_c, t_rule=rule, cv_error=cv_error,
                        train_marginal_error=residual)
            selected_train_dist = train_dist

    train_kernel, smallest, shift = make_psd(np.exp(-selected_train_dist / best["t"]))
    model = fit_svm(train_kernel, y_train, best["C_svm"])
    # Stream test rows to keep the test-by-training matrix from getting huge.
    predictions = []
    test_residual = 0.0
    for start in range(0, len(X_test), 64):
        test_dist, residual = distance_matrix(
            X_test[start:start + 64], X_train, M, best["lambda_ot"], MAX_ITER
        )
        test_kernel = np.exp(-test_dist / best["t"])
        predictions.extend(predict_svm(model, test_kernel).tolist())
        test_residual = max(test_residual, residual)
    predictions = np.asarray(predictions, dtype=int)
    best.update(test_error=float(np.mean(predictions != y_test)),
                smallest_eigenvalue=smallest, diagonal_shift=shift,
                test_marginal_error=test_residual,
                n_train=len(X_train), n_test=len(X_test))
    return best, predictions


def main():
    images, labels = load_mnist(find_idx("train-images.idx3-ubyte"),
                                find_idx("train-labels.idx1-ubyte"))
    if max(N_VALUES) > len(images):
        raise ValueError("Requested more images than the input file contains.")
    # Exactly the same indices are used for images AND their labels.
    selected = np.random.default_rng(SEED).permutation(len(images))[:max(N_VALUES)]
    normalized_histogram = make_histograms(images[selected])
    selected_labels = labels[selected]
    M = make_cost_matrix()
    results = []
    config = dict(full_experiment=FULL_EXPERIMENT, N_values=N_VALUES, seed=SEED,
                  max_iter=MAX_ITER, side=SIDE, resize="BICUBIC",
                  lambda_numerators=LAMBDA_NUMERATORS, t_rules=T_RULES,
                  c_values=C_VALUES, inner_repeats=INNER_REPEATS,
                  outer_repeats=OUTER_REPEATS, outer_folds_to_run=OUTER_FOLDS_TO_RUN)
    for N in N_VALUES:
        X, y = normalized_histogram[:N], selected_labels[:N]
        for repeat in range(OUTER_REPEATS):
            folds = stratified_folds(y, 4, SEED + repeat)
            for fold in range(OUTER_FOLDS_TO_RUN):
                train_indices = folds[fold]
                test_indices = np.concatenate([folds[k] for k in range(4) if k != fold])
                print(f"N={N}, repeat={repeat + 1}, fold={fold + 1}", flush=True)
                result, prediction = evaluate_split(
                    X[train_indices], y[train_indices], X[test_indices], y[test_indices],
                    M, SEED + 100 * repeat + 10 * fold
                )
                result.update(N=N, repeat=repeat + 1, fold=fold + 1,
                              original_train_indices=selected[train_indices].tolist(),
                              original_test_indices=selected[test_indices].tolist(),
                              true_labels=y[test_indices].tolist(),
                              predictions=prediction.tolist())
                results.append(result)
                OUTPUT_FILE.write_text(json.dumps(dict(config=config, runs=results), indent=2))
                print(f"  t={result['t']:.4g}, C={result['C_svm']}, "
                      f"test error={100 * result['test_error']:.2f}%", flush=True)
        errors = [r["test_error"] for r in results if r["N"] == N]
        print(f"N={N}: mean error = {100 * np.mean(errors):.2f}% over {len(errors)} run(s)")
        if len(errors) > 1:
            print(f"Sample standard deviation = {100 * np.std(errors, ddof=1):.2f} percentage points")
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
