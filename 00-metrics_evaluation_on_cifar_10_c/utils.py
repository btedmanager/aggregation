import os
import csv
import torch
# =========================
# Experiment Configuration
# =========================

# -------- Federated Setup --------
NUM_CLIENTS = 20
NUM_CLIENTS_AT_ONCE = 1
NUM_ROUNDS = 20
TOP_K_CLIENTS = NUM_CLIENTS // 2
SEED = 42

# -------- Training Hyperparameters --------
LOCAL_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LOSS_FN = "CrossEntropyLoss"

# -------- Dataset Configuration --------
CLEAN_CLIENTS = NUM_CLIENTS // 2
CORRUPTED_CLIENTS = NUM_CLIENTS // 2

DATASET_NAME_CLEAN = "CIFAR-10"
DATASET_NAME_CORRUPTED = "CIFAR-10-C"

CIFAR10_ROOT = "./data"
CIFAR10C_ROOT = "./CIFAR-10-C"

CIFAR10C_CORRUPTION = os.getenv("CIFAR10C_CORRUPTION", "brightness")
CIFAR10C_SEVERITY = None

NUM_CLASSES = 10

# -------- Model Configuration --------
INPUT_CHANNELS = 3
IMAGE_SIZE = 32

# -------- Logging --------
LOG_DIR = f"./logs/{CIFAR10C_CORRUPTION}"

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
    "auc", "loss", "tp", "tn", "fp", "fn", "round_time",
    "number_noisy_clients_selected", "number_clean_clients_selected"
]

# -------- Runtime --------
# Allow forcing CPU via environment variable
if os.getenv("FORCE_CPU", "0") == "1":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE_NAME = "CPU"

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

# =========================
# Data Utilities
# =========================
def load_npy_file(file_path, mmap_mode='r'):
    """
    Safely load a .npy file. By default uses memory mapping to handle large files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    try:
        import numpy as np
        return np.load(file_path, mmap_mode=mmap_mode)
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")

def set_seed(seed: int = SEED):
    """
    Set seeds for reproducibility.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
