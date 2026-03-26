import os
import torch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from LSTM import LSTM_Baseline  # Reuse Member 2's model definition for 100% consistency

# Fix random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Dataset Loader Class: Exactly same as baseline
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
        # Return sample ID for result statistics
        return idx, features, label

if __name__ == "__main__":
    # -------------------------- INPUT DATA PATHS --------------------------
    test_tfidf_path = r'C:\Users\86180\Desktop\X_test_tfidf.npz'
    test_label_path = r'C:\Users\86180\Desktop\y_test.npy'
    best_model_path = r'C:\Users\86180\Desktop\best_model.pth'

    # -------------------------- OUTPUT PATH (FIXED TO YOUR TARGET FOLDER) --------------------------
    target_root = r'C:\Users\86180\Desktop\member4_experiment2'
    output_table_root = os.path.join(target_root, 'output_table')
    os.makedirs(output_table_root, exist_ok=True)

    device = torch.device("cpu")

    # Load test set
    test_dataset = TFIDFDataset(test_tfidf_path, test_label_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load baseline LSTM model
    model = LSTM_Baseline(input_dim=5000, hidden_dim=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Initialize prediction result list (matches instructor's required fields)
    predict_result = []

    # Predict ONLY the first 100 samples
    with torch.no_grad():
        for i, (sample_id, features, labels) in enumerate(test_loader):
            if i >= 100:
                break
            sample_id = sample_id.item()
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            # Calculate prediction probability, label and confidence
            prob = torch.softmax(outputs, dim=1)
            predict_confidence, predicted_label = torch.max(prob, 1)
            true_label = labels.item()
            predicted_label = predicted_label.item()
            predict_confidence = predict_confidence.item()
            # Extract TF-IDF feature vector
            tfidf_feature = features.squeeze().cpu().numpy()

            # Save to result list
            predict_result.append({
                'Sample_ID': sample_id,
                'Text_TFIDF_Feature_Vector': tfidf_feature,
                'True_Label_(0=Real_News/1=Fake_News)': true_label,
                'Predicted_Label_(0=Real_News/1=Fake_News)': predicted_label,
                'Prediction_Confidence': round(predict_confidence, 4),
                'Prediction_Correct': 'Yes' if predicted_label == true_label else 'No'
            })

    # Convert to DataFrame and save
    predict_df = pd.DataFrame(predict_result)
    predict_df.to_excel(os.path.join(output_table_root, 'TestSet_Top100_Prediction_Results.xlsx'), index=False)
    # Also save CSV format for Member 5's visualization
    predict_df.to_csv(os.path.join(output_table_root, 'TestSet_Top100_Prediction_Results.csv'), index=False)

    # Calculate prediction accuracy
    top100_acc = predict_df['Prediction_Correct'].value_counts(normalize=True)['Yes']
    print(f"===== Top 100 Test Set Samples Prediction Completed! =====")
    print(f"Prediction Accuracy: {top100_acc:.4f}")
    print(f"Results saved to: {output_table_root}")
