import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
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

# Feature selection
selector = SelectKBest(f_classif, k=10)  # Reduced to top 10 features for efficiency
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected Features:", selected_features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [10, 20, None],  # Maximum depth of trees
    'min_samples_split': [2, 5]   # Minimum samples to split a node
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("\nEnhanced Random Forest Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
model_output = 'models/rf_model.pkl'
joblib.dump(best_rf_model, model_output)
print(f"Model saved to {model_output}")