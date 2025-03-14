import torch
from torch_geometric.data import Data, DataEdgeAttr
import torch.serialization
import pandas as pd
from train_gnn import GNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data with safe globals
try:
    with torch.serialization.safe_globals([Data, DataEdgeAttr]):
        data = torch.load('data/DDos_dataset_processed.pt')
    logging.info(f"Loaded graph data from 'data/DDos_dataset_processed.pt' with {data.num_nodes} nodes")
except FileNotFoundError:
    logging.error("Error: 'data/DDos_dataset_processed.pt' not found!")
    exit()
except Exception as e:
    logging.error(f"Error loading graph data: {e}")
    exit()

# Check if test_mask exists and is properly defined
if not hasattr(data, 'test_mask') or data.test_mask.sum() == 0:
    logging.error("Test mask is missing or empty. Ensure the dataset was preprocessed with train/test splits.")
    exit()

# Move data to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
data = data.to(device)

# Load model
try:
    model = GNN(num_features=data.num_features, hidden_channels=16, num_classes=2).to(device)
    state_dict = torch.load('models/gnn_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("GNN model loaded successfully.")
except FileNotFoundError:
    logging.error("Error: GNN model file 'models/gnn_model.pt' not found!")
    exit()
except Exception as e:
    logging.error(f"Error loading GNN model: {e}")
    exit()

# Evaluation
try:
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        y_test = data.y[data.test_mask]
        y_pred = pred[data.test_mask]
        y_true = y_test

        # Compute metrics
        metrics = {
            'Model': ['GNN'],
            'Accuracy': [accuracy_score(y_true.cpu(), y_pred.cpu())],
            'Precision': [precision_score(y_true.cpu(), y_pred.cpu())],
            'Recall': [recall_score(y_true.cpu(), y_pred.cpu())],
            'F1-Score': [f1_score(y_true.cpu(), y_pred.cpu())]
        }
        metrics_df = pd.DataFrame(metrics)
        print("\nEvaluation Metrics:")
        print(metrics_df)

        # Confusion Matrix
        cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
        plt.title('Confusion Matrix - GNN')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('results/gnn_confusion_matrix.png')
        logging.info("Confusion matrix saved to 'results/gnn_confusion_matrix.png'")
        plt.close()

except Exception as e:
    logging.error(f"Error during evaluation: {e}")
    exit()

if __name__ == "__main__":
    print("Evaluation complete.")