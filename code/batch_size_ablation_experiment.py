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

# Fix all random seeds for full reproducibility, exactly same as Member 2's baseline
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# LSTM Model: EXACTLY THE SAME AS MEMBER 2'S BASELINE, DO NOT MODIFY
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
        lstm_out, (hn, cn) = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Early Stopping Class: EXACTLY THE SAME AS MEMBER 2'S BASELINE, DO NOT MODIFY
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Dataset Loader Class: EXACTLY THE SAME AS MEMBER 2'S BASELINE, DO NOT MODIFY
class TFIDFDataset(Dataset):
    def __init__(self, tfidf_path, label_path):
        npz_file = np.load(tfidf_path, allow_pickle=True)
        sparse_mat = csr_matrix(
            (npz_file['data'], npz_file['indices'], npz_file['indptr']),
            shape=tuple(npz_file['shape'])
        )
        self.tfidf = sparse_mat.toarray()
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        features = torch.tensor(self.tfidf[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        features = features.unsqueeze(0)
        return features, label

# Training Function for Single Batch Size Experiment
def run_single_experiment(batch_size, dataset_paths, result_root, model_root):
    # Create dedicated folders for current batch size to avoid data overwriting
    result_dir = os.path.join(result_root, f'batch_size_{batch_size}')
    model_dir = os.path.join(model_root, f'batch_size_{batch_size}')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'best_model.pth')

    device = torch.device("cpu")
    print(f"===== Starting Experiment: batch_size = {batch_size}, Device: {device} =====")

    # Load dataset from your desktop (Member 1's data)
    train_dataset = TFIDFDataset(dataset_paths['train_tfidf'], dataset_paths['train_label'])
    val_dataset = TFIDFDataset(dataset_paths['val_tfidf'], dataset_paths['val_label'])
    test_dataset = TFIDFDataset(dataset_paths['test_tfidf'], dataset_paths['test_label'])

    # ONLY BATCH_SIZE IS CHANGED HERE, all other configs same as baseline
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, Loss Function: 100% same as Member 2's baseline
    model = LSTM_Baseline(input_dim=5000, hidden_dim=64, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5, delta=0, path=model_save_path)

    # Initialize training log
    log_df = pd.DataFrame(columns=['epoch', 'batch_size', 'train_loss', 'train_acc', 'val_acc'])
    epochs_max = 50

    # Training Loop (exact same as Member 2's baseline)
    for epoch in range(1, epochs_max + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs_max} | batch_size={batch_size}')

        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

        # Calculate average training metrics
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation process
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total

        # Print epoch results
        print(f'Epoch {epoch} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        # Save single epoch log
        log_df.loc[epoch - 1] = [epoch, batch_size, train_loss_avg, train_acc, val_acc]
        log_df.to_csv(os.path.join(result_dir, 'train_log.csv'), index=False)

        # Early stopping check
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f'Early Stopping Triggered. Best Validation Accuracy: {early_stopping.best_score:.4f}')
            break

    # Test set final evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = test_correct / test_total

    # Save final complete log with test accuracy
    log_df['test_acc'] = test_acc
    log_df.to_csv(os.path.join(result_dir, 'train_log_final.csv'), index=False)
    print(f'===== Experiment Completed: batch_size={batch_size} | Best Val Acc: {early_stopping.best_score:.4f} | Test Acc: {test_acc:.4f} =====\n')

    return log_df

# Main Function: Run all 5 batch size experiments
if __name__ == "__main__":
    # -------------------------- INPUT DATA PATHS (MEMBER 1'S DATA ON YOUR DESKTOP) --------------------------
    dataset_paths = {
        'train_tfidf': r'C:\Users\86180\Desktop\X_train_tfidf.npz',
        'train_label': r'C:\Users\86180\Desktop\y_train.npy',
        'val_tfidf': r'C:\Users\86180\Desktop\X_val_tfidf.npz',
        'val_label': r'C:\Users\86180\Desktop\y_val.npy',
        'test_tfidf': r'C:\Users\86180\Desktop\X_test_tfidf.npz',
        'test_label': r'C:\Users\86180\Desktop\y_test.npy'
    }

    # -------------------------- OUTPUT PATH (FIXED TO YOUR TARGET FOLDER) --------------------------
    # All results will be saved here, no more missing files
    target_root = r'C:\Users\86180\Desktop\member4_experiment2'
    result_root = os.path.join(target_root, 'result')  # Training logs for each batch size
    model_root = os.path.join(target_root, 'model')    # Model weights for each batch size
    output_table_root = os.path.join(target_root, 'output_table')  # Final summary Excel

    # Auto create all folders, no manual setup needed
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(output_table_root, exist_ok=True)

    # 5 batch sizes specified by the instructor
    batch_size_list = [8, 16, 32, 64, 128]
    # Store all experiment data
    all_experiment_data = pd.DataFrame()

    # Run each experiment in loop
    for bs in batch_size_list:
        single_log = run_single_experiment(bs, dataset_paths, result_root, model_root)
        all_experiment_data = pd.concat([all_experiment_data, single_log], ignore_index=True)

    # Save standardized raw data (format aligned with Member 3)
    all_experiment_data.to_excel(os.path.join(output_table_root, 'Experiment2_BatchSize_Comparison_RawData.xlsx'), index=False)
    print("===== All Batch Size Experiments Completed! =====")
    print(f"===== All results are saved to: {target_root} =====")