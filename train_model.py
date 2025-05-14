import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load UCI dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')

# X = features, y = target
X = data.drop(['name', 'status'], axis=1)
y = data['status']
print(X.shape)   # after feature selection
print(y.shape)   # after feature selection

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save model + scaler
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
