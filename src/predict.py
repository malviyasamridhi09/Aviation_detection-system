import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/maintenance_model.pkl")

# New sensor data (example)
input_data = pd.DataFrame([{
    "temperature": 82,
    "pressure": 106,
    "vibration": 0.05
}])

# Predict
prediction = model.predict(input_data)

print("\n🛫 Prediction Result:")
if prediction[0] == 1:
    print("⚠️ Maintenance Required")
else:
    print("✅ Aircraft Operating Normally")
