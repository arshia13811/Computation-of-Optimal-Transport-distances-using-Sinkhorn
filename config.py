import os
from pathlib import Path

# Prevent numpy from hogging all CPU cores during distance calculations
os.environ["OMP_NUM_THREADS"] = "1"

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "MNIST"
OUTPUT_FILE = BASE_DIR / "sinkhorn_results.json"

# Experiment toggle
FULL_EXPERIMENT = False

# Image and algorithm settings
SIDE = 20
MAX_ITER = 20
BATCH_SIZE = 256
SEED = 42

# Grid search and sizing parameters
if FULL_EXPERIMENT:
    N_VALUES = [3000, 5000, 12000, 17000, 25000]
    OUTER_REPEATS = 6
    OUTER_FOLDS_TO_RUN = 4
    LAMBDA_NUMERATORS = [5, 7, 9, 11]
else:
    N_VALUES = [400]
    OUTER_REPEATS = 1
    OUTER_FOLDS_TO_RUN = 1
    LAMBDA_NUMERATORS = [9]

T_RULES = ["one", "q10", "q20", "q50"]
C_VALUES = [0.01, 1.0, 10000.0]
INNER_REPEATS = 2
