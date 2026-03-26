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


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LSTM
class LSTM_Baseline(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.7):
        super(LSTM_Baseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(1)

        # LSTM forward propagation
        lstm_out, (hn, cn) = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


# Early stopping
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


# Load dataset
class TFIDFDataset(Dataset):
    def __init__(self, tfidf_path, label_path):
        # Restore to a dense matrix
        npz_file = np.load(tfidf_path, allow_pickle=True)
        sparse_mat = csr_matrix(
            (npz_file['data'], npz_file['indices'], npz_file['indptr']),
            shape=tuple(npz_file['shape'])
        )
        self.tfidf = sparse_mat.toarray()

        # Load label file
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.tfidf)

    def __getitem__(self, idx):
        features = torch.tensor(self.tfidf[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        features = features.unsqueeze(0)
        return features, label


# Train
def train_model():
    device = torch.device("cpu")
    print(f"The equipment used is {device}")

    # Load data
    train_dataset = TFIDFDataset(
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\X_train_tfidf.npz',
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\y_train.npy'
    )
    val_dataset = TFIDFDataset(
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\X_val_tfidf.npz',
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\y_val.npy'
    )
    test_dataset = TFIDFDataset(
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\X_test_tfidf.npz',
        r'F:\1hkm\LU - AIBA\课程\T2\CDS 525 - Deep Learning\Group Project\M1\member1_outputs (1)\y_test.npy'
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = LSTM_Baseline(input_dim=5000, hidden_dim=64, num_layers=2).to(device)

    # Optimizer and loss function (learning rate = 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay = 5e-5)  # 1e-4
    criterion = nn.CrossEntropyLoss()

    # Early stopping (patience = 5)
    early_stopping = EarlyStopping(patience=5, delta=0, path='best_model.pth')

    # Save log
    os.makedirs('result', exist_ok=True)
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_acc'])

    # Training
    epochs_max = 50
    for epoch in range(1, epochs_max + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs_max}')
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

        # Calculate training indicates
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
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

        print(
            f'Epoch {epoch} | Loss of training：{train_loss_avg:.4f} | Training accuracy：{train_acc:.4f} | Validation accuracy：{val_acc:.4f}')

        # Save log
        log_df.loc[epoch - 1] = [epoch, train_loss_avg, train_acc, val_acc]
        log_df.to_csv('result/train_log.csv', index=False)

        # Early termination judgment
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f'Early stopping, optimal validation set accuracy rate：{early_stopping.best_score:.4f}')
            break

    # Testing
    model.load_state_dict(torch.load('best_model.pth'))
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

    # Save final log
    log_df['test_acc'] = test_acc
    log_df.to_csv('result/train_log_final.csv', index=False)

    # Print final results
    print('=' * 50)
    print(
        f'Base model training completed | Optimal validation set accuracy: {early_stopping.best_score:.4f} | Accuracy of the test set: {test_acc:.4f}')
    print('=' * 50)


if __name__ == "__main__":
    train_model()