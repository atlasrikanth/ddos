import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load preprocessed data
data_file = 'data/DDos_dataset_processed.csv'
try:
    df = pd.read_csv(data_file)
    print(f"Loaded data from {data_file}")
    print("Columns in dataset:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: File '{data_file}' not found!")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Determine the correct target column
target_col = 'Label' if 'Label' in df.columns else 'label' if 'label' in df.columns else None
if target_col:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print(f"Using target column: {target_col}")
else:
    print("Error: Target column 'Label' or 'label' not found in dataset!")
    exit()

# Validate and clean data
print("Shape of X before cleaning:", X.shape)
print("Sample of X head:\n", X.head())
print("Data types of X columns:\n", X.dtypes)
print("Checking for NaN or infinite values...")
if X.isnull().any().any() or np.isinf(X).any().any():
    print("Warning: NaN or infinite values detected. Replacing with 0.")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
print("Shape of X after cleaning:", X.shape)

# Ensure y is binary and valid
y = y.astype(int)
print("Unique values in y:", np.unique(y))
if len(np.unique(y)) != 2:
    print(f"Error: y must contain exactly 2 unique values (binary classification). Found: {np.unique(y)}")
    print("Please check your dataset or preprocessing steps to ensure both 'Normal' and 'DDoS' classes are present.")
    exit()
if not np.all(np.isin(y, [0, 1])):
    print("Error: y contains values other than 0 or 1. Converting to binary...")
    y = (y > 0).astype(int)
    print("New unique values in y:", np.unique(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)
print("y_train unique values:", np.unique(y_train))
print("y_test unique values:", np.unique(y_test))

# Manual feature and target check
expected_features = 10  # Based on your pipeline
if X_train.shape[1] != expected_features:
    print(f"Error: Expected {expected_features} features, but got {X_train.shape[1]}. Columns: {X.columns.tolist()}")
    exit()
if X_train.shape[0] != y_train.shape[0]:
    print(f"Error: Mismatch between X_train rows ({X_train.shape[0]}) and y_train rows ({y_train.shape[0]})")
    exit()

# Train Random Forest
try:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Fitting Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("Random Forest model fitted successfully.")
    y_pred_rf = rf_model.predict(X_test)
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'DDoS']))
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("Random Forest model saved to models/rf_model.pkl")
except Exception as e:
    print(f"Error training Random Forest: {e}")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_train head:\n", X_train.head())
    print("y_train head:\n", y_train.head())
    print("X_train dtypes:\n", X_train.dtypes)
    exit()

# Train Deep Learning Model
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.summary()

# Add EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
try:
    dl_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
except Exception as e:
    print(f"Error training Deep Learning model: {e}")
    exit()

# Evaluate Deep Learning Model
y_pred_dl_proba = dl_model.predict(X_test, verbose=0)
y_pred_dl = (y_pred_dl_proba > 0.5).astype(int).flatten()
print("\nDeep Learning Performance:")
print(classification_report(y_test, y_pred_dl, target_names=['Normal', 'DDoS']))

# Save the model
dl_model.save('models/dl_model.keras')
print("Deep Learning model saved to models/dl_model.keras")