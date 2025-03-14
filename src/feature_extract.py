from scapy.all import *
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(packets, duration=10):
    """
    Extract features from a list of packets and create a graph structure for GNN.
    Args:
        packets: List of packets captured by Scapy.
        duration: Duration of capture in seconds.
    Returns:
        PyG Data object with features and graph structure.
    """
    if not packets:
        logging.warning("No packets provided to extract_features.")
        return None
    
    # Initialize variables
    flow_duration = duration * 1000000  # Convert to microseconds
    node_features_list = []
    edge_index = []
    src_ips = set()
    dst_ips = set()
    
    # Use the first packet to determine flow direction
    try:
        start_time = packets[0].time
        src_ip = packets[0][IP].src if IP in packets[0] else None
        dst_ip = packets[0][IP].dst if IP in packets[0] else None
        logging.info(f"Flow direction set: Source IP = {src_ip}, Destination IP = {dst_ip}")
    except (IndexError, KeyError, AttributeError) as e:
        logging.error(f"Error determining flow direction: {e}")
        return None
    
    last_fwd_time = None
    last_bwd_time = None
    valid_packet_count = 0
    
    for i, pkt in enumerate(packets):
        if IP not in pkt:
            logging.debug(f"Packet {i} has no IP layer. Skipping.")
            continue
        valid_packet_count += 1
        
        # Add to graph nodes (one node per packet)
        src_ips.add(pkt[IP].src)
        dst_ips.add(pkt[IP].dst)
        packet_length = len(pkt)
        
        # Basic features per packet
        features = {
            'Flow Duration': flow_duration,
            'Total Fwd Packets': 1 if pkt[IP].src == src_ip else 0,
            'Total Backward Packets': 1 if pkt[IP].dst == src_ip else 0,
            'Fwd Packet Length Mean': packet_length if pkt[IP].src == src_ip else 0,
            'Bwd Packet Length Mean': packet_length if pkt[IP].dst == src_ip else 0,
            'Flow Bytes/s': packet_length / (duration if duration > 0 else 1),
            'Flow Packets/s': 1 / (duration if duration > 0 else 1),
            'Fwd IAT Mean': pkt.time - start_time if pkt[IP].src == src_ip and last_fwd_time else 0,
            'Bwd IAT Mean': pkt.time - start_time if pkt[IP].dst == src_ip and last_bwd_time else 0,
            'Packet Length Mean': packet_length
        }
        node_features_list.append([features[col] for col in [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
            'Flow Packets/s', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean'
        ]])
        
        # Update IAT tracking
        if pkt[IP].src == src_ip and last_fwd_time:
            features['Fwd IAT Mean'] = pkt.time - last_fwd_time
        if pkt[IP].dst == src_ip and last_bwd_time:
            features['Bwd IAT Mean'] = pkt.time - last_bwd_time
        if pkt[IP].src == src_ip:
            last_fwd_time = pkt.time
        if pkt[IP].dst == src_ip:
            last_bwd_time = pkt.time
        
        # Add edges based on communication
        if i > 0 and (pkt[IP].src == packets[i-1][IP].dst or pkt[IP].dst == packets[i-1][IP].src):
            edge_index.append([i-1, i])
            edge_index.append([i, i-1])
    
    logging.info(f"Processed {valid_packet_count} valid packets: Fwd = {sum(f[1] for f in node_features_list)}, Bwd = {sum(f[2] for f in node_features_list)}")
    if not node_features_list:
        logging.error("No valid packet features extracted.")
        return None
    
    # Convert to DataFrame and scale
    df = pd.DataFrame(node_features_list, columns=[
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
        'Flow Packets/s', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean'
    ])
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = scaler.transform(df)
    except FileNotFoundError:
        logging.error("Scaler file 'models/scaler.pkl' not found!")
        return None
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        return None
    
    # Create graph structure
    edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(X_scaled, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    return data

if __name__ == "__main__":
    # Capture more packets for a meaningful graph
    logging.info("Capturing traffic for 30 seconds...")
    packets = sniff(timeout=30, filter="ip")
    if not packets:
        logging.warning("No packets captured during the 30-second window.")
    else:
        logging.info(f"Captured {len(packets)} packets.")
        graph = extract_features(packets, duration=30)
        if graph is not None:
            print(graph)
        else:
            print("Failed to create graph.")