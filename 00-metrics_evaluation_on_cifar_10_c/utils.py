import os
import csv
# =========================
# Experiment Configuration
# =========================

# -------- Federated Setup --------
NUM_CLIENTS = 10
NUM_CLIENTS_AT_ONCE = 1
NUM_ROUNDS = 10
TOP_K_CLIENTS = NUM_CLIENTS // 2

# -------- Training Hyperparameters --------
LOCAL_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LOSS_FN = "CrossEntropyLoss"

# -------- Dataset Configuration --------
CLEAN_CLIENTS = NUM_CLIENTS // 2
CORRUPTED_CLIENTS = NUM_CLIENTS // 2

DATASET_NAME_CLEAN = "CIFAR-10"
DATASET_NAME_CORRUPTED = "CIFAR-10-C"

CIFAR10_ROOT = "./data"
CIFAR10C_ROOT = "./CIFAR-10-C"

CIFAR10C_CORRUPTION = "gaussian_noise"
CIFAR10C_SEVERITY = None

NUM_CLASSES = 10

# -------- Model Configuration --------
INPUT_CHANNELS = 3
IMAGE_SIZE = 32

# -------- Logging --------
LOG_DIR = "./logs"

CLIENTS_METADATA_FILE = f"{LOG_DIR}/client_metadata.csv"
CLIENT_METRICS_FILE = f"{LOG_DIR}/client_metrics.csv"
GLOBAL_METRICS_FILE = f"{LOG_DIR}/global_metrics.csv"
CLIENT_EVAL_METRICS_FILE = f"{LOG_DIR}/client_eval_metrics.csv"
EXPERIMENT_PARAMS_FILE = f"{LOG_DIR}/experiment_params.txt"

# -------- Evaluation --------
EVAL_BATCH_SIZE = 64


CLIENT_EVAL_METRICS_HEADER = [
    "round", "cid", "dataset_type","accuracy", "precision", "recall", 
    "f1_score", "AUC", "avg_loss", "TP", "TN", "FP", "FN"
]
GLOBAL_METRICS_HEADER = [
    "round", "accuracy", "precision", "recall", "f1_score",
    "auc", "loss", "tp", "tn", "fp", "fn", "round_time"
]

# -------- Runtime --------
DEVICE = "cpu"  # will be resolved dynamically

# =========================
# Save Experiment Parameters
# =========================
def save_experiment_config():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(EXPERIMENT_PARAMS_FILE, "w") as f:
        for k, v in globals().items():
            if k.isupper():
                f.write(f"{k}: {v}\n")
    print(f"✅ Experiment Config saved to {EXPERIMENT_PARAMS_FILE}")


def create_global_metrics_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(GLOBAL_METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GLOBAL_METRICS_HEADER)
    print(f"✅ Global Metrics saved to {GLOBAL_METRICS_FILE}")

def create_eval_metrics_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CLIENT_EVAL_METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CLIENT_EVAL_METRICS_HEADER)
    print(f"✅ Client Evaluation Metrics saved to {CLIENT_EVAL_METRICS_FILE}")

