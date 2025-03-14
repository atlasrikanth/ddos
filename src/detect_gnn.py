from scapy.all import *
from feature_extract import extract_features
import torch
import joblib
import logging
from train_gnn import GNN

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and scaler with error handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

try:
    model = GNN(num_features=10, hidden_channels=16, num_classes=2).to(device)
    # Load model state dict with map_location to ensure device compatibility
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

try:
    scaler = joblib.load('models/scaler.pkl')
    logging.info("Scaler loaded successfully.")
except FileNotFoundError:
    logging.error("Error: Scaler file 'models/scaler.pkl' not found!")
    exit()
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    exit()

def process_traffic(packets):
    # Extract features and create graph
    graph = extract_features(packets, duration=10)
    if graph is None:
        logging.warning("No valid packets captured or feature extraction failed.")
        return "No packets captured or feature extraction failed."
    
    # Verify the number of features
    if graph.num_features != 10:
        logging.error(f"Feature mismatch: Expected 10 features, got {graph.num_features}")
        return f"Feature mismatch: Expected 10 features, got {graph.num_features}"
    
    # Debug graph structure
    logging.info(f"Graph structure: num_nodes={graph.num_nodes}, num_edges={graph.edge_index.shape[1]}")
    
    # Move graph to device
    graph = graph.to(device)
    
    # Make prediction
    try:
        with torch.no_grad():
            out = model(graph)
            logging.info(f"Model output shape: {out.shape}")
            if out.shape[0] == 0:
                return "No valid predictions from model."
            
            # Handle single or multiple nodes
            if out.shape[0] == 1:  # Single node graph
                probs = torch.softmax(out, dim=1)
                dl_pred_proba = probs[0, 1].cpu().numpy()  # Probability of DDoS
            else:  # Multiple nodes, average probability
                probs = torch.softmax(out, dim=1)
                dl_pred_proba = probs[:, 1].mean().cpu().numpy()  # Average probability of DDoS
            
            dl_pred = 1 if dl_pred_proba > 0.5 else 0
    except Exception as e:
        logging.error(f"Error during GNN prediction: {e}")
        return f"Prediction error: {e}"
    
    result = {
        'GNN': 'DDoS' if dl_pred == 1 else 'Normal',
        'GNN Probability': float(dl_pred_proba)
    }
    logging.info(f"Detection Result: {result}")
    if dl_pred == 1:
        logging.warning("Potential DDoS attack detected!")
    return result

def capture_and_detect():
    logging.info("Capturing traffic for 10 seconds...")
    try:
        packets = sniff(timeout=10, filter="ip")
        if not packets:
            logging.warning("No packets captured during the 10-second window.")
            return
        logging.info(f"Captured {len(packets)} packets.")
        result = process_traffic(packets)
        print("Detection Result:", result)
    except Exception as e:
        logging.error(f"Error during packet capture: {e}")
        print("Error: Packet capture failed. Ensure Npcap is installed and run as Administrator.")

if __name__ == "__main__":
    capture_and_detect()