import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Fix random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 1. Path Configuration =====================
BASE_INPUT_PATH = "../01_Input_Data"
BASE_OUTPUT_PATH = "../03_Results/M3_Experiments"

os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
BEST_MODEL_DIR = os.path.join(BASE_OUTPUT_PATH, "best_models")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

TRAIN_TFIDF_PATH = os.path.join(BASE_INPUT_PATH, "X_train_tfidf.npz")
TRAIN_LABEL_PATH = os.path.join(BASE_INPUT_PATH, "y_train.npy")
VAL_TFIDF_PATH = os.path.join(BASE_INPUT_PATH, "X_val_tfidf.npz")
VAL_LABEL_PATH = os.path.join(BASE_INPUT_PATH, "y_val.npy")
TEST_TFIDF_PATH = os.path.join(BASE_INPUT_PATH, "X_test_tfidf.npz")
TEST_LABEL_PATH = os.path.join(BASE_INPUT_PATH, "y_test.npy")

METRICS_SAVE_PATH = os.path.join(BASE_OUTPUT_PATH, "member3_all_metrics.csv")

# ===================== 2. Model Definition =====================
class LSTM_Baseline(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.7):
        super(LSTM_Baseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ===================== 3. Focal Loss Definition =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ===================== 4. Early Stopping =====================
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_acc = 0.0

    def __call__(self, val_acc, model, save_path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            self.counter = 0

# ===================== 5. Dataset =====================
class TFIDFDataset(Dataset):
    def __init__(self, tfidf_path, label_path):
        npz_file = np.load(tfidf_path, allow_pickle=True)
        mat = csr_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']), shape=tuple(npz_file['shape']))
        self.tfidf = mat.toarray()
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        feat = torch.tensor(self.tfidf[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

# ===================== 6. Training Pipeline =====================
def run_experiment(model, device, train_loader, val_loader, test_loader,
                   lr, criterion_name, epochs_max=30, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if criterion_name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "focal":
        criterion = FocalLoss()
    else:
        raise ValueError("Criterion not supported")

    es = EarlyStopping(patience=patience)
    model_path = os.path.join(BEST_MODEL_DIR, f"best_{criterion_name}_lr_{lr}.pth")
    metrics = []

    for epoch in range(1, epochs_max + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"LR={lr}, Loss={criterion_name} | Epoch {epoch}")
        for feat, y in pbar:
            feat, y = feat.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(feat)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feat.size(0)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            pbar.set_postfix({"loss": loss.item()})

        train_loss = total_loss / total
        train_acc = correct / total

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feat, y in val_loader:
                feat, y = feat.to(device), y.to(device)
                pred = model(feat)
                _, predicted = torch.max(pred, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        val_acc = val_correct / val_total

        # Test
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for feat, y in test_loader:
                feat, y = feat.to(device), y.to(device)
                pred = model(feat)
                _, predicted = torch.max(pred, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
        test_acc = test_correct / test_total

        print(f"Epoch {epoch:2d} | TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} | TestAcc {test_acc:.4f}")
        metrics.append({
            "epoch": epoch,
            "learning_rate": lr,
            "loss_function": criterion_name,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4)
        })

        es(val_acc, model, model_path)
        if es.early_stop:
            print(f"Early stop triggered. Best Val Acc: {es.best_val_acc:.4f}\n")
            break

    return metrics

# ===================== 7. Main =====================
if __name__ == "__main__":
    device = torch.device("cpu")
    print("Using device:", device)

    # Load data
    train_loader = DataLoader(TFIDFDataset(TRAIN_TFIDF_PATH, TRAIN_LABEL_PATH), batch_size=32, shuffle=True)
    val_loader = DataLoader(TFIDFDataset(VAL_TFIDF_PATH, VAL_LABEL_PATH), batch_size=32, shuffle=False)
    test_loader = DataLoader(TFIDFDataset(TEST_TFIDF_PATH, TEST_LABEL_PATH), batch_size=32, shuffle=False)

    all_results = []

    # ===================== Experiment 1: Learning Rate Tuning =====================
    lr_list = [0.1, 0.01, 0.001, 0.0005]
    print("\n========== START LEARNING RATE TUNING ==========\n")
    for lr in lr_list:
        print(f"Now training with LR = {lr}")
        model = LSTM_Baseline().to(device)
        res = run_experiment(model, device, train_loader, val_loader, test_loader,
                             lr=lr, criterion_name="ce")
        all_results.extend(res)

    # ===================== Experiment 2: Loss Function Comparison =====================
    print("\n========== START LOSS FUNCTION COMPARISON ==========\n")
    loss_functions = ["ce", "focal"]
    for loss_fn in loss_functions:
        print(f"Now training with Loss = {loss_fn} (fixed LR=0.001)")
        model = LSTM_Baseline().to(device)
        res = run_experiment(model, device, train_loader, val_loader, test_loader,
                             lr=0.001, criterion_name=loss_fn)
        all_results.extend(res)

    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(METRICS_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\nAll experiments done! Results saved to: {METRICS_SAVE_PATH}")