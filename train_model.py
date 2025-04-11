import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta

# Generate simulated smartwatch-style data
n_samples = 500
timestamps = [datetime.now() - timedelta(minutes=i * 10) for i in range(n_samples)]

data = {
    'Timestamp': timestamps,
    'Heart_Rate_bpm': np.random.randint(60, 100, size=n_samples),
    'SpO2_%': np.random.normal(97, 1, size=n_samples).round(1),
    'Blood_Pressure_Systolic': np.random.randint(110, 140, size=n_samples),
    'Blood_Pressure_Diastolic': np.random.randint(70, 90, size=n_samples),
    'Sleep_Duration_hr': np.random.normal(7.5, 1, size=n_samples).round(2),
    'Steps_Count': np.random.randint(0, 5000, size=n_samples),
    'Calories_Burned': np.random.randint(1500, 3000, size=n_samples),
    'Stress_Level': np.random.randint(1, 10, size=n_samples),
    'HRV_ms': np.random.normal(60, 15, size=n_samples).round(1),
    'Skin_Temperature_C': np.random.normal(36.5, 0.3, size=n_samples).round(1),
    'Respiration_Rate_bpm': np.random.randint(12, 20, size=n_samples),
    'VO2_Max': np.random.normal(40, 5, size=n_samples).round(1),
}

df = pd.DataFrame(data)

# Simulate heart risk label (0 = low risk, 1 = high risk)
df['Heart_Risk'] = ((df['Heart_Rate_bpm'] > 90) |
                    (df['SpO2_%'] < 95) |
                    (df['Stress_Level'] > 7) |
                    (df['Sleep_Duration_hr'] < 6)).astype(int)

# Save simulated data
df.to_csv("simulated_smartwatch_data.csv", index=False)

# Train model
X = df.drop(columns=['Timestamp', 'Heart_Risk'])
y = df['Heart_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "smartwatch_heart_model.pkl")

print("✅ Model trained and saved as 'smartwatch_heart_model.pkl'")
print("Training Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
