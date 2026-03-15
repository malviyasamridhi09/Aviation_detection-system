# Load dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
data = pd.read_csv("datasets/sensor_data.csv")

# Split features and target
X = data.drop("failure", axis=1)
y = data["failure"]
##print("Training features:", X.columns.tolist())


