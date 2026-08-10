"""
Shared configuration for the TriageNet pipeline.

Keeping labels, paths and hyperparameters in one place means every script
(train.py, evaluate.py, explainability.py) agrees on the same class order,
which matters a lot for confusion matrices and AUC-ROC to be interpretable.
"""
from pathlib import Path

# --- Labels -----------------------------------------------------------
LABELS = ["Normal", "Needs Attention", "Urgent"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABELS)
URGENT_IDX = LABEL2IDX["Urgent"]

# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# --- Image branch ---------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Text branch ---------------------------------------------------------
# Bio_ClinicalBERT is trained on MIMIC-III notes; BioBERT (dmis-lab) is trained
# on PubMed/PMC. Either is a reasonable choice per the proposal -- default here,
# override with --text-model-name on the CLI if your group prefers the other.
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_TOKEN_LEN = 256

# --- Training defaults ---------------------------------------------------
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-5
RANDOM_SEED = 42
