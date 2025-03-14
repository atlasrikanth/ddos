import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # Added missing import

# Load data with error handling
try:
    df = pd.read_csv('data/cicids2017_processed.csv')
    print("Data loaded successfully from 'data/cicids2017_processed.csv'")
except FileNotFoundError:
    print("Error: 'data/cicids2017_processed.csv' not found!")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Prepare data
X = df.drop('Label', axis=1)
y = df['Label']
if 'Label' not in df.columns:
    print("Error: 'Label' column not found in the dataset!")
    exit()

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models with error handling
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
    dl_model = tf.keras.models.load_model('models/dl_model.keras')  # Updated to .keras
    print("Deep Learning model loaded successfully.")
except FileNotFoundError:
    print("Error: Deep Learning model file 'models/dl_model.keras' not found!")
    exit()
except Exception as e:
    print(f"Error loading Deep Learning model: {e}")
    exit()

# Predictions
try:
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest predictions completed.")
except Exception as e:
    print(f"Error with Random Forest prediction: {e}")
    exit()

try:
    y_pred_dl = (dl_model.predict(X_test) > 0.5).astype(int)
    print("Deep Learning predictions completed.")
except Exception as e:
    print(f"Error with Deep Learning prediction: {e}")
    exit()

# Evaluation metrics
metrics = {
    'Model': ['Random Forest', 'Deep Learning'],
    'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_dl)],
    'Precision': [precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_dl)],
    'Recall': [recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_dl)],
    'F1-Score': [f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_dl)]
}
metrics_df = pd.DataFrame(metrics)
print("\nEvaluation Metrics:")
print(metrics_df)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for Deep Learning
cm_dl = confusion_matrix(y_test, y_pred_dl)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('Confusion Matrix - Deep Learning')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()