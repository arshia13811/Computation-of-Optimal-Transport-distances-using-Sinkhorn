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
FULL_EXPERIMENT = False
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

