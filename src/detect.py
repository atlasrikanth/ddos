from scapy.all import *
import pandas as pd
import joblib
import tensorflow as tf
from feature_extract import extract_features

# Load models and scaler
try:
    rf_model = joblib.load('models/rf_model.pkl')
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print("Error: Random Forest model file 'models/rf_model.pkl' not found!")
    exit()
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    exit()

try:
    dl_model = tf.keras.models.load_model('models/dl_model.keras')
    print("Deep Learning model loaded successfully.")
except FileNotFoundError:
    print("Error: Deep Learning model file 'models/dl_model.keras' not found!")
    exit()
except Exception as e:
    print(f"Error loading Deep Learning model: {e}")
    exit()

try:
    scaler = joblib.load('models/scaler.pkl')
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Scaler file 'models/scaler.pkl' not found!")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

def process_traffic(packets):
    # Extract features using the imported function
    try:
        features = extract_features(packets, duration=10)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        print("Please check 'feature_extract.py' to ensure the 'extract_features' function accepts a 'duration' parameter.")
        return "Feature extraction failed."
    
    if features is None or features.empty:
        return "No packets captured or feature extraction failed."
    
    # Ensure features match the expected columns
    expected_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
        'Flow Packets/s', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean'
    ]
    if list(features.columns) != expected_columns:
        print(f"Error: Extracted features do not match expected columns. Got {list(features.columns)}")
        return "Feature mismatch error."
    
    # Scale the features while preserving column names
    try:
        features_scaled = scaler.transform(features)
        # Convert back to DataFrame to preserve column names for RandomForest
        features_scaled_df = pd.DataFrame(features_scaled, columns=expected_columns)
    except Exception as e:
        print(f"Error scaling features: {e}")
        return "Feature scaling error."
    
    # Make predictions
    try:
        rf_pred = rf_model.predict(features_scaled_df)[0]  # Use DataFrame with column names
    except Exception as e:
        print(f"Error with Random Forest prediction: {e}")
        return "Random Forest prediction error."
    
    try:
        dl_pred_proba = dl_model.predict(features_scaled, verbose=0)[0][0]
        dl_pred = 1 if dl_pred_proba > 0.5 else 0
    except Exception as e:
        print(f"Error with Deep Learning prediction: {e}")
        return "Deep Learning prediction error."
    
    result = {
        'Random Forest': 'DDoS' if rf_pred == 1 else 'Normal',
        'Deep Learning': 'DDoS' if dl_pred == 1 else 'Normal',
        'DL Probability': float(dl_pred_proba)
    }
    return result

def capture_and_detect():
    print("Capturing traffic for 10 seconds...")
    try:
        # Try to capture packets at Layer 2
        packets = sniff(timeout=10)
        if not packets:
            print("No packets captured during the 10-second window.")
            return
        print(f"Captured {len(packets)} packets.")
        result = process_traffic(packets)
        print("Detection Result:", result)
    except Exception as e:
        print(f"Error during packet capture: {e}")
        print("This may be due to missing Npcap/WinPcap or lack of Administrator privileges.")
        print("Please ensure Npcap is installed (https://npcap.com/) and run this script as Administrator.")
        print("Attempting to use Layer 3 socket as a fallback...")
        try:
            # Fallback to Layer 3 socket
            conf.L3socket = conf.L3socket  # Ensure Layer 3 socket is used
            packets = sniff(timeout=10)
            if not packets:
                print("No packets captured even with Layer 3 socket.")
                return
            print(f"Captured {len(packets)} packets using Layer 3 socket.")
            result = process_traffic(packets)
            print("Detection Result:", result)
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            print("Please ensure Npcap is installed and run this script with Administrator privileges.")

if __name__ == "__main__":
    capture_and_detect()