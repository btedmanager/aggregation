# server.py
import flwr as fl
from strategy import ScoreWeightedFedAvg
from utils import NUM_ROUNDS, NUM_CLIENTS

strategy = ScoreWeightedFedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS
)

fl.server.start_server(
    server_address="127.0.0.1:5000",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)