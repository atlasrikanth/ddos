import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data_file = 'data/DDos_dataset.csv'  # Use your preprocessed dataset
try:
    df = pd.read_csv(data_file)
    print(f"Loaded data from {data_file}")
    print("Columns in dataset:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: File '{data_file}' not found!")
    exit()

# Features and target (adjusted to 'label' lowercase)
target_col = 'label'  # Matches your dataset
if target_col in df.columns:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
else:
    print(f"Error: Target column '{target_col}' not found in dataset!")
    exit()

# Ensure y is binary (0 or 1) and convert to numpy array
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.summary()  # Print model architecture

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = dl_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
y_pred_proba = dl_model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Flatten to match y_test shape

print("\nDeep Learning Model Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model (using .keras format)
model_output = 'models/dl_model.keras'
dl_model.save(model_output)
print(f"Model saved to {model_output}")