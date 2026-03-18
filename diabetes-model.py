import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load Data
# Dataset: Pima Indians Diabetes Database
data = pd.read_csv('diabetes.csv')

print("----- DATASET STATISTICS -----")
print(data.describe())

# 2. Define Features and Target
y = data.Outcome
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[features]

print("\n----- FEATURES PREVIEW (X) -----")
print(X.head())

# 3. Split data into Training and Validation sets (80/20 split)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# 4. Optimization: Testing Decision Tree size to find best max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

print("\n----- DECISION TREE SIZE TESTING -----")
for max_leaves in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaves, train_X, val_X, train_y, val_y)
    print(f"Max leaf nodes: {max_leaves} \t MAE: {my_mae:.6f}")

# 5. Final Model: Random Forest Regressor
print("\n----- TRAINING RANDOM FOREST MODEL -----")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

# 6. Prediction and Evaluation
forest_preds = forest_model.predict(val_X)
forest_mae = mean_absolute_error(val_y, forest_preds)

print("\n----- FINAL RESULTS -----")
print(f"Random Forest MAE: {forest_mae:.6f}")

# Convert continuous predictions to binary (threshold 0.5)
hard_preds = [1 if x > 0.5 else 0 for x in forest_preds]

# Confusion Matrix for diagnostic accuracy
cm = confusion_matrix(val_y, hard_preds)
print("\n----- CONFUSION MATRIX (Diagnostic Results) -----")
print(f"True Negatives (Healthy): {cm[0][0]}")
print(f"False Positives (False Alarm): {cm[0][1]}")
print(f"False Negatives (Missed Cases): {cm[1][0]}")
print(f"True Positives (Detected Diabetes): {cm[1][1]}")

# 7. Feature Importance Ranking
importances = pd.Series(forest_model.feature_importances_, index=features)
print("\n--- FEATURE IMPORTANCE RANKING ---")
print(importances.sort_values(ascending=False))
