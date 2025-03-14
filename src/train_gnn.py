import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train():
    # Load preprocessed graph data with weights_only=False
    try:
        data = torch.load('data/DDos_dataset_processed.pt', weights_only=False)
        logging.info(f"Loaded graph data from 'data/DDos_dataset_processed.pt' with {data.num_nodes} nodes")
    except FileNotFoundError:
        logging.error("Error: 'data/DDos_dataset_processed.pt' not found!")
        return
    except Exception as e:
        logging.error(f"Error loading graph data: {e}")
        return
    
    # Check if labels exist
    if not hasattr(data, 'y') or data.y is None:
        logging.error("Graph data has no labels (y). Ensure preprocessing includes labels.")
        return
    
    # Split data into train, validation, and test sets
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_idx = perm[:int(0.7 * num_nodes)]  # 70% for training
    val_idx = perm[int(0.7 * num_nodes):int(0.9 * num_nodes)]  # 20% for validation
    test_idx = perm[int(0.9 * num_nodes):]  # 10% for testing
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model = GNN(num_features=data.num_features, hidden_channels=16, num_classes=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    # Training loop with validation
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    model.train()
    
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_correct = (val_pred == data.y[data.val_mask]).sum()
            val_acc = val_correct / data.val_mask.sum()
        
        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/gnn_model.pt')
            logging.info("Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_out = model(data)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_correct = (test_pred == data.y[data.test_mask]).sum()
        test_acc = test_correct / data.test_mask.sum()
        logging.info(f'Test Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    train()