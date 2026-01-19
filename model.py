from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import numpy as np  # Ensure numpy is imported

# Load preprocessed data splits
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_dropout_train = joblib.load('y_dropout_train.pkl')
y_dropout_test = joblib.load('y_dropout_test.pkl')
y_stress_train = joblib.load('y_stress_train.pkl')
y_stress_test = joblib.load('y_stress_test.pkl')

# Train dropout classifier
dropout_model = RandomForestClassifier(n_estimators=100, random_state=42)
dropout_model.fit(X_train, y_dropout_train)
dropout_pred = dropout_model.predict(X_test)
print(f"Dropout Accuracy: {accuracy_score(y_dropout_test, dropout_pred):.2f}")

# Train stress regressor
stress_model = RandomForestRegressor(n_estimators=100, random_state=42)
stress_model.fit(X_train, y_stress_train)
stress_pred = stress_model.predict(X_test)
# Compute RMSE manually (compatible with all scikit-learn versions)
rmse = np.sqrt(mean_squared_error(y_stress_test, stress_pred))
print(f"Stress RMSE: {rmse:.2f}")

# Save models
joblib.dump(dropout_model, 'dropout_model.pkl')
joblib.dump(stress_model, 'stress_model.pkl')

print("Model training complete. Models saved.")