import flwr as fl
from flwr.common import FitIns, EvaluateIns
import time
import csv
from utils import GLOBAL_METRICS_FILE, TOP_K_CLIENTS

class ScoreWeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        scored_clients = []

        for _, fit_res in results:
            score = fit_res.metrics["mean_cumulative_loss"]
            params = fit_res.parameters
            scored_clients.append((params, score))

        top_clients = sorted(scored_clients, key=lambda x: x[1])[:TOP_K_CLIENTS]

        total_weight = sum(score for _, score in top_clients)

        aggregated = []
        for layer in zip(*[fl.common.parameters_to_ndarrays(p) for p, _ in top_clients]):
            aggregated.append(
                sum(score * l for l, (_, score) in zip(layer, top_clients)) / total_weight
            )

        return fl.common.ndarrays_to_parameters(aggregated), {"round": rnd}
    
    def aggregate_evaluate(self, rnd, results, failures):
        start_time = time.time()

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
        loss = weighted_avg("loss") if "loss" in results[0][1].metrics else 0.0

        tp = sum(res.metrics["tp"] for _, res in results)
        tn = sum(res.metrics["tn"] for _, res in results)
        fp = sum(res.metrics["fp"] for _, res in results)
        fn = sum(res.metrics["fn"] for _, res in results)

        round_time = time.time() - start_time

        # ----- Log global metrics -----
        with open(GLOBAL_METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                rnd, acc, prec, rec, f1, auc, loss,
                tp, tn, fp, fn, round_time
            ])

        return loss, {
            "accuracy": float(acc),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc
        }

    def configure_fit(self, server_round, parameters, client_manager):
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