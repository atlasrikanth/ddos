import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import joblib

def preprocess_to_graph(input_file, output_file):
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}")
        print("Original columns:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: {input_file} not found! Generating synthetic dataset...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=['src_bytes', 'dst_bytes', 'duration', 'packet_count', 'protocol_type', 'flag', 'service', 'land', 'wrong_fragment', 'urgent'])
        df['label'] = y
        df.to_csv(input_file, index=False)
        print(f"Synthetic dataset saved to {input_file}")
    
    # Handle missing values and infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Map DDos_dataset.csv features to CICIDS2017-like features
    features_df = pd.DataFrame()
    features_df['Flow Duration'] = df['duration'] * 1000000  # Convert seconds to microseconds
    features_df['Total Fwd Packets'] = df['packet_count'] // 2  # Approximate: split packet_count evenly
    features_df['Total Backward Packets'] = df['packet_count'] - features_df['Total Fwd Packets']
    features_df['Fwd Packet Length Mean'] = df['src_bytes'] / features_df['Total Fwd Packets'].replace(0, 1)
    features_df['Bwd Packet Length Mean'] = df['dst_bytes'] / features_df['Total Backward Packets'].replace(0, 1)
    features_df['Flow Bytes/s'] = (df['src_bytes'] + df['dst_bytes']) / df['duration'].replace(0, 1)
    features_df['Flow Packets/s'] = df['packet_count'] / df['duration'].replace(0, 1)
    features_df['Fwd IAT Mean'] = df['duration'] / features_df['Total Fwd Packets'].replace(0, 1)
    features_df['Bwd IAT Mean'] = df['duration'] / features_df['Total Backward Packets'].replace(0, 1)
    features_df['Packet Length Mean'] = (df['src_bytes'] + df['dst_bytes']) / df['packet_count'].replace(0, 1)
    
    # Extract features and labels
    X = features_df
    y = df['label'].apply(lambda x: 1 if any(kw in str(x).upper() for kw in ['DDOS', 'ATTACK', 'MALICIOUS', '1', 'ANOMALY']) else 0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to 'models/scaler.pkl'")
    
    # Create graph structure
    # Since DDos_dataset.csv doesn't have Source IP/Destination IP, we'll connect consecutive rows
    # Alternatively, we can use 'protocol_type' or 'service' to define edges
    edge_index = []
    for i in range(len(df) - 1):
        # Connect consecutive rows (simplified approach)
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
        # Optional: Add edges based on protocol_type or service
        if df.iloc[i]['protocol_type'] == df.iloc[i + 1]['protocol_type']:
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Convert to PyG Data object
    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Save graph data
    torch.save(data, output_file)
    print(f"Graph data saved to {output_file}")

if __name__ == "__main__":
    try:
        preprocess_to_graph('data/DDos_dataset.csv', 'data/DDos_dataset_processed.pt')
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        exit()