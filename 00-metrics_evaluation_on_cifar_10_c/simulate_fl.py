import shutil, os
import flwr as fl
from client import FLClient
from strategy import ScoreWeightedFedAvg
from utils import NUM_CLIENTS, NUM_CLIENTS_AT_ONCE, NUM_ROUNDS, CLEAN_CLIENTS
from utils import save_experiment_config, create_global_metrics_file, create_eval_metrics_file
from utils import LOG_DIR, CLIENTS_METADATA_FILE
import numpy as np
from dataset import load_dataset
from client_metadata import save_clients_metadata_csv, extract_client_metadata_from_loader
import torch
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


shutil.rmtree(LOG_DIR) if os.path.exists(LOG_DIR) else print("FOLDER DOESN'T EXIST")
save_experiment_config()
create_global_metrics_file()
create_eval_metrics_file()


# ----------------------
# Client factory
# ----------------------
def client_fn(cid: str):
    """
    Create a FLClient with the given client id.
    """
    return FLClient(int(cid)).to_client()


# ----------------------
# Strategy
# ----------------------
strategy = ScoreWeightedFedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS_AT_ONCE,
    min_available_clients=NUM_CLIENTS
)


def save_client_metadata():
    # ----------------------
    # Save client metadata BEFORE simulation
    # ----------------------
    clients_metadata = []

    for cid in range(NUM_CLIENTS):
        dataloader = load_dataset(cid)
        # Extract all labels from the DataLoader
        labels = []
        for _, y in dataloader:
            labels.extend(y.numpy())
        labels = np.array(labels)

        # Determine if the client is clean
        is_clean = cid < CLEAN_CLIENTS

        # Then call the function correctly
        meta = extract_client_metadata_from_loader(
            cid=cid,
            labels=labels,
            is_clean=is_clean,
            num_classes=10,
        )


        clients_metadata.append(meta)

    save_clients_metadata_csv(
        clients_metadata,
        output_path=CLIENTS_METADATA_FILE,
    )

    print("✅ Client metadata saved to clients_metadata.csv")

save_client_metadata()


# ----------------------
# Start Simulation
# ----------------------
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    client_resources = {"num_cpus": 1, "num_gpus": 0},
    strategy=strategy
)
