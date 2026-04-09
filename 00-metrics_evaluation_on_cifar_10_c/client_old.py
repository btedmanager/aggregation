import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
from model import SimpleCNN
from dataset import load_dataset
from utils import (
    LOCAL_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    CLIENT_EVAL_METRICS_FILE,
    CLEAN_CLIENTS
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from collections import OrderedDict




def parameters_to_state_dict(model, parameters):
    state_dict = OrderedDict()
    for key, param in zip(model.state_dict().keys(), parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict



class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.dataset_type = "CIFAR-10" if cid < CLEAN_CLIENTS else "CIFAR-10-C"
        self.model = SimpleCNN().to(DEVICE)
        self.trainloader = load_dataset(cid)
        self.valloader = load_dataset(cid)  # or separate val split
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    # -----------------
    # Training
    # -----------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_cumulative_loss = 0.0
        start_time = time.time()

        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            epoch_cumulative_loss = epoch_loss / len(self.trainloader.dataset)
            total_cumulative_loss += epoch_cumulative_loss

        # Mean cumulative loss over epochs
        mean_cum_loss = total_cumulative_loss / LOCAL_EPOCHS

        # Return updated parameters + metrics
        return self.get_parameters(), len(self.trainloader.dataset), {"mean_cumulative_loss": float(mean_cum_loss)}

    # -----------------
    # Evaluation
    # -----------------
    def evaluate(self, parameters, config):
        round_number = config.get("round", -1)
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred, y_prob = [], [], []
        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                total_loss += loss.item() * x.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc = 0.0
        # cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        # tp = cm.trace()
        # fp = cm.sum() - tp
        # fn = cm.sum() - tp
        # tn = cm.sum() - (tp + fp + fn)
        import numpy as np
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        tp_arr = np.diag(cm)
        fp_arr = cm.sum(axis=0) - tp_arr
        fn_arr = cm.sum(axis=1) - tp_arr
        tn_arr = cm.sum() - (fp_arr + fn_arr + tp_arr)

        tp = int(tp_arr.sum())
        fp = int(fp_arr.sum())
        fn = int(fn_arr.sum())
        tn = int(tn_arr.sum())
        
        avg_loss = total_loss / len(self.valloader.dataset)

        # Log metrics
        with open(CLIENT_EVAL_METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_number, self.cid, self.dataset_type,
                acc, prec, rec, f1, auc, avg_loss, tp, tn, fp, fn
            ])

        return float(avg_loss), len(self.valloader.dataset), {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc": float(auc),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn)
        }
    
    # -----------------
    # Helper functions
    # -----------------
    def get_parameters(self, config=None):
        """
        Return model parameters as list of NumPy arrays.
        `config` is required by Flower Simulation API, even if unused.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        state_dict = parameters_to_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
    
    # def set_parameters(self, parameters):
    #     params_dict = zip(self.model.state_dict().keys(), parameters)
    #     state_dict = {k: torch.tensor(v) for k, v in params_dict}
    #     self.model.load_state_dict(state_dict, strict=True)
