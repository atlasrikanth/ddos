import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_data(input_file, output_file):
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}")
        print("Columns in dataset:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        return
    
    # Handle missing values and infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Only fill numeric columns
    
    # Encode categorical features (not needed for CICIDS2017-like features)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Label']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Encoded categorical column: {col}")
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    
    # Select features (all columns except 'Label')
    features = [col for col in df.columns if col != 'Label']
    if len(features) != 10:
        print(f"Warning: Expected 10 features, but found {len(features)}: {features}")
    
    X = df[features]
    # Adjust target column to 'Label'
    if 'Label' in df.columns:
        y = df['Label']
    else:
        print("Error: Target column 'Label' not found in dataset!")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler for live prediction
    
    # Save preprocessed data
    processed_df = pd.DataFrame(X_scaled, columns=features)
    processed_df['Label'] = y
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Use the extracted features
    input_file = 'data/DDos_extracted_features.csv'
    output_file = 'data/DDos_dataset_processed.csv'
    preprocess_data(input_file, output_file)