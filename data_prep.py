import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
data = {
    'grades': np.random.normal(70, 15, n_samples).clip(0, 100),
    'attendance': np.random.normal(80, 10, n_samples).clip(0, 100),
    'hours_studied': np.random.normal(5, 2, n_samples).clip(0, 20),
    'social_interactions': np.random.normal(5, 2, n_samples).clip(0, 10),
    'stress_score': np.random.normal(5, 2, n_samples).clip(0, 10),
    'dropout': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% dropout rate
}
df = pd.DataFrame(data)

# Preprocessing
features = ['grades', 'attendance', 'hours_studied', 'social_interactions', 'stress_score']
X = df[features]
y_dropout = df['dropout']
y_stress = df['stress_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_dropout_train, y_dropout_test, y_stress_train, y_stress_test = train_test_split(
    X_scaled, y_dropout, y_stress, test_size=0.2, random_state=42
)

# Save scaler and splits for use in other scripts
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_dropout_train, 'y_dropout_train.pkl')
joblib.dump(y_dropout_test, 'y_dropout_test.pkl')
joblib.dump(y_stress_train, 'y_stress_train.pkl')
joblib.dump(y_stress_test, 'y_stress_test.pkl')

print("Data preparation complete. Files saved.")