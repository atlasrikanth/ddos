import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate synthetic data (skip if using real dataset)
X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, n_classes=2, random_state=42)
columns = ['src_bytes', 'dst_bytes', 'duration', 'packet_count', 'protocol_type', 'flag', 'service', 'land', 'wrong_fragment', 'urgent']
data = pd.DataFrame(X, columns=columns)
data['label'] = y
data.to_csv('data/DDos_dataset.csv', index=False)

# Load data
data = pd.read_csv('data/DDos_dataset.csv')
X = data.drop('label', axis=1)
y = data['label']

# Step 2: Baseline Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Baseline Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
joblib.dump(rf_model, 'models/rf_model.pkl')

# Step 3: Enhanced Random Forest
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_enhanced_rf = best_rf_model.predict(X_test)
print("Enhanced Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_enhanced_rf):.4f}")
joblib.dump(best_rf_model, 'models/rf_model.pkl')

# Step 4: Deep Learning Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
y_pred_proba = dl_model.predict(X_test)
y_pred_dl = (y_pred_proba > 0.5).astype(int)
print("Deep Learning Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dl):.4f}")
dl_model.save('models/dl_model.h5')

# Step 5: Visualization
cm = confusion_matrix(y_test, y_pred_dl)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('Confusion Matrix - Deep Learning Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()