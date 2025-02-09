# This code uses Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
PATH = 'Crop_recommendation.csv'
data = pd.read_csv(PATH)

# Prepare features and labels
# Assuming 'label' is the target column and others are features
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']

# Encode labels if they are categorical
labels = pd.get_dummies(labels).values

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the values for the test dataset
predicted_values = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test.argmax(axis=1), predicted_values.argmax(axis=1))
print('Test accuracy:', accuracy)