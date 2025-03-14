import pandas as pd
from sklearn.datasets import make_classification

data_file = 'data/DDos_dataset.csv'
try:
    df = pd.read_csv(data_file)
    print(f"Loaded data from {data_file}")
    print("Columns in dataset:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: File '{data_file}' not found!")
    # Generate synthetic data
    X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=['src_bytes', 'dst_bytes', 'duration', 'packet_count', 'protocol_type', 'flag', 'service', 'land', 'wrong_fragment', 'urgent'])
    df['label'] = y
    df['label'] = df['label'].map({0: 'normal', 1: 'attack'})
    df.to_csv('data/DDos_dataset.csv', index=False)
    print("Generated synthetic dataset with both classes.")
    print("Columns in dataset:", df.columns.tolist())

if 'label' in df.columns:
    print("Unique values in 'label':", df['label'].unique())
    print("Label distribution:\n", df['label'].value_counts())
    y = df['label'].apply(lambda x: 1 if any(kw in str(x).upper() for kw in ['DDOS', 'ATTACK', 'MALICIOUS', '1']) else 0)
    print("Binary label distribution (0: Normal, 1: DDoS/Attack/Malicious):\n", y.value_counts())
else:
    print("Error: 'label' column not found!")