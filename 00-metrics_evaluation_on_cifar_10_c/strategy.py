import flwr as fl
from flwr.common import FitIns, EvaluateIns
import time
import csv
from utils import GLOBAL_METRICS_FILE, TOP_K_CLIENTS, CLEAN_CLIENTS

class ScoreWeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        scored_clients = []

        for client_proxy, fit_res in results:
            score = fit_res.metrics["mean_cumulative_loss"]
            params = fit_res.parameters
            scored_clients.append((client_proxy, params, score))

        # Select top-K clients based on lowest score (loss)
        top_clients = sorted(scored_clients, key=lambda x: x[2])[:TOP_K_CLIENTS]

        # Count noisy and clean clients among the selected top-K
        self.num_clean_selected = 0
        self.num_noisy_selected = 0
        for client_proxy, _, _ in top_clients:
            cid = int(client_proxy.cid)
            if cid < CLEAN_CLIENTS:
                self.num_clean_selected += 1
            else:
                self.num_noisy_selected += 1

        total_weight = sum(score for _, _, score in top_clients)

        aggregated = []
        for layer in zip(*[fl.common.parameters_to_ndarrays(p) for _, p, _ in top_clients]):
            aggregated.append(
                sum(score * l for l, (_, _, score) in zip(layer, top_clients)) / total_weight
            )

        return fl.common.ndarrays_to_parameters(aggregated), {"round": rnd}
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return 0.0, {}

        total_samples = sum(res.num_examples for _, res in results)

        def weighted_avg(metric):
            return sum(
                res.num_examples * res.metrics[metric]
                for _, res in results
            ) / total_samples

        acc = weighted_avg("accuracy")
        prec = weighted_avg("precision")
        rec = weighted_avg("recall")
        f1 = weighted_avg("f1_score")
        auc = weighted_avg("auc")
        loss = weighted_avg("loss") if results and "loss" in results[0][1].metrics else 0.0

        tp = sum(res.metrics["tp"] for _, res in results)
        tn = sum(res.metrics["tn"] for _, res in results)
        fp = sum(res.metrics["fp"] for _, res in results)
        fn = sum(res.metrics["fn"] for _, res in results)

        # Get selected counts (fallback to 0 if fit hasn't run yet)
        num_noisy = getattr(self, "num_noisy_selected", 0)
        num_clean = getattr(self, "num_clean_selected", 0)

        # Calculate time since the start of the round (configure_fit)
        if hasattr(self, "round_start_time"):
            round_time = time.time() - self.round_start_time
        else:
            round_time = 0.0

        # ----- Log global metrics -----
        with open(GLOBAL_METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                rnd, acc, prec, rec, f1, auc, loss,
                tp, tn, fp, fn, round_time,
                num_noisy, num_clean
            ])

        return loss, {
            "accuracy": float(acc),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc
        }

    def configure_fit(self, server_round, parameters, client_manager):
        # Start the timer for the round
        self.round_start_time = time.time()
        clients = list(client_manager.all().values())
        config = {"round": server_round}
        # send FitIns(parameters, config) to each client
        return [(client, FitIns(parameters, config)) for client in clients]

    # -----------------
    # Evaluate configuration
    # -----------------
    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = list(client_manager.all().values())
        config = {"round": server_round}  # <- this is key
        # send EvaluateIns(parameters, config) to each client
        return [(client, EvaluateIns(parameters, config)) for client in clients]