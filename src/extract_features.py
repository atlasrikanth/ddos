import pandas as pd
import numpy as np

def extract_features(df):
    """
    Transform features from DDos_dataset.csv to match CICIDS2017-like features.
    Input: DataFrame with columns ['src_bytes', 'dst_bytes', 'duration', 'packet_count', 'protocol_type', 'flag', 'service', 'land', 'wrong_fragment', 'urgent', 'label']
    Output: DataFrame with CICIDS2017-like features.
    """
    if df.empty:
        print("Input DataFrame is empty!")
        return None
    
    # Initialize the features DataFrame
    features = pd.DataFrame()
    
    # Map DDos_dataset.csv features to CICIDS2017-like features
    features['Flow Duration'] = df['duration'] * 1000000  # Convert seconds to microseconds
    features['Total Fwd Packets'] = df['packet_count'] // 2  # Approximate: split packet_count evenly
    features['Total Backward Packets'] = df['packet_count'] - features['Total Fwd Packets']
    features['Fwd Packet Length Mean'] = df['src_bytes'] / features['Total Fwd Packets'].replace(0, 1)  # Avoid division by zero
    features['Bwd Packet Length Mean'] = df['dst_bytes'] / features['Total Backward Packets'].replace(0, 1)
    features['Flow Bytes/s'] = (df['src_bytes'] + df['dst_bytes']) / df['duration'].replace(0, 1)
    features['Flow Packets/s'] = df['packet_count'] / df['duration'].replace(0, 1)
    
    # Approximate IAT (Inter-Arrival Time) assuming uniform distribution of packets
    features['Fwd IAT Mean'] = df['duration'] / features['Total Fwd Packets'].replace(0, 1)
    features['Bwd IAT Mean'] = df['duration'] / features['Total Backward Packets'].replace(0, 1)
    
    # Packet Length Mean (average of src_bytes and dst_bytes per packet)
    features['Packet Length Mean'] = (df['src_bytes'] + df['dst_bytes']) / df['packet_count'].replace(0, 1)
    
    # Add the label (handle different label formats)
    if 'label' in df.columns:
        print("Unique values in 'label' before transformation:", df['label'].unique())
        # Try multiple formats for DDoS/attack labels
        features['Label'] = df['label'].apply(lambda x: 1 if any(kw in str(x).upper() for kw in ['DDOS', 'ATTACK', 'MALICIOUS', '1', 'ANOMALY']) else 0)
        print("Binary label distribution after transformation (0: Normal, 1: DDoS/Attack/Malicious):\n", features['Label'].value_counts())
        if len(features['Label'].unique()) != 2:
            print("Error: After transformation, 'Label' must contain exactly 2 unique values. Found:", features['Label'].unique())
            return None
    else:
        print("Error: 'label' column not found!")
        return None
    
    return features

if __name__ == "__main__":
    # Load your dataset
    try:
        df = pd.read_csv('data/DDos_dataset.csv')
        print(f"Loaded data from data/DDos_dataset.csv")
        print("Original columns:", df.columns.tolist())
    except FileNotFoundError:
        print("Error: 'data/DDos_dataset.csv' not found!")
        exit()
    
    # Extract features
    features_df = extract_features(df)
    if features_df is not None:
        print("\nExtracted features:")
        print(features_df.head())
        
        # Save the extracted features to a new CSV for use in your pipeline
        output_file = 'data/DDos_extracted_features.csv'
        features_df.to_csv(output_file, index=False)
        print(f"\nExtracted features saved to {output_file}")
    else:
        print("Feature extraction failed due to invalid label distribution.")