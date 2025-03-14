from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from feature_extract import extract_features
from scapy.all import sniff
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='ddos_detection.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load models and scaler with error handling
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Capturing traffic for 10 seconds...")
    try:
        packets = sniff(timeout=10, filter="ip")
        if not packets:
            logging.info("No packets captured during the 10-second window.")
            return jsonify({'error': 'No packets captured during the 10-second window.'})
        print(f"Captured {len(packets)} packets.")
        
        features = extract_features(packets, duration=10)
        if features is None or features.empty:
            logging.info("No valid features extracted from captured packets.")
            return jsonify({'error': 'No valid features extracted from captured packets.'})
        
        expected_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
            'Flow Packets/s', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean'
        ]
        if list(features.columns) != expected_columns:
            logging.error(f"Extracted features do not match expected columns. Got {list(features.columns)}")
            return jsonify({'error': f'Extracted features do not match expected columns. Got {list(features.columns)}'})
        
        try:
            features_scaled = scaler.transform(features)
            features_scaled_df = pd.DataFrame(features_scaled, columns=expected_columns)
        except Exception as e:
            logging.error(f"Error scaling features: {e}")
            return jsonify({'error': f'Error scaling features: {e}'})
        
        try:
            rf_pred = rf_model.predict(features_scaled_df)[0]
        except Exception as e:
            logging.error(f"Error with Random Forest prediction: {e}")
            return jsonify({'error': f'Error with Random Forest prediction: {e}'})
        
        try:
            dl_pred_proba = dl_model.predict(features_scaled, verbose=0)[0][0]
            dl_pred = 1 if dl_pred_proba > 0.5 else 0
        except Exception as e:
            logging.error(f"Error with Deep Learning prediction: {e}")
            return jsonify({'error': f'Error with Deep Learning prediction: {e}'})
        
        result = {
            'Random Forest': 'DDoS' if rf_pred == 1 else 'Normal',
            'Deep Learning': 'DDoS' if dl_pred == 1 else 'Normal',
            'DL Probability': float(dl_pred_proba)
        }
        logging.info(f"Detection Result: {result}")
        if rf_pred == 1 or dl_pred == 1:
            logging.warning("Potential DDoS attack detected!")
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error during packet capture: {e}")
        return jsonify({'error': f'Error during packet capture: {e}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)