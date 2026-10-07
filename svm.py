import numpy as np
from scipy.linalg import eigh
from libsvm.svmutil import svm_parameter, svm_predict, svm_problem, svm_train


def make_psd(train_kernel, margin=1e-8):
    S = np.asarray(train_kernel, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.isfinite(S).all():
        raise ValueError("Training kernel must be finite and square.")
    # Fixed Sinkhorn iterations can leave small directional asymmetries.
    asymmetry = float(np.max(np.abs(S - S.T)))
    S = 0.5 * (S + S.T)
    smallest = float(eigh(S, subset_by_index=[0, 0], eigvals_only=True)[0])
    shift = max(0.0, margin - smallest)
    S.flat[::len(S) + 1] += shift
    return S, {"minimum_eigenvalue_before_shift": smallest,
               "diagonal_shift": shift, "kernel_asymmetry": asymmetry}


def _libsvm_rows(kernel):
    # LIBSVM precomputed format needs a leading 1-based sample serial number.
    S = np.asarray(kernel, dtype=np.float64)
    if S.ndim != 2 or not np.isfinite(S).all():
        raise ValueError("Kernel must be a finite 2D array.")
    serial = np.arange(1, len(S) + 1, dtype=float)[:, None]
    return np.ascontiguousarray(np.concatenate((serial, S), axis=1))


def fit_precomputed(train_kernel, y_train, C_svm=1.0):
    S = np.asarray(train_kernel)
    y_train = np.asarray(y_train)
    if S.shape != (len(y_train), len(y_train)):
        raise ValueError("Expected an n_train x n_train kernel and n_train labels.")
    if len(np.unique(y_train)) < 2 or C_svm <= 0:
        raise ValueError("Need at least two classes and positive C_svm.")
    problem = svm_problem(y_train, _libsvm_rows(S), isKernel=True)
    # -s 0 = C-SVC, -t 4 = precomputed kernel, -c = SVM penalty.
    parameter = svm_parameter(f"-s 0 -t 4 -c {float(C_svm)} -e 0.001 -q")
    model = svm_train(problem, parameter)
    return model, len(y_train)


def predict_precomputed(fitted, test_kernel):
    model, n_train = fitted
    S = np.asarray(test_kernel)
    if S.ndim != 2 or S.shape[1] != n_train:
        raise ValueError("Test kernel must have one column per TRAINING image.")
    # Dummy labels are only for LIBSVM's unused metrics; they do not affect prediction.
    labels, _, _ = svm_predict(np.zeros(len(S)), _libsvm_rows(S), model, "-q")
    return np.asarray(labels, dtype=int)
