import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
PATH = 'Crop_recommendation.csv'
data = pd.read_csv(PATH)

# Prepare features and labels for Logistic Regression
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']

# Logistic Regression
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
LogReg = LogisticRegression(random_state=42).fit(X_train, Y_train)
y_pred_log = LogReg.predict(X_test)

# Collect Logistic Regression metrics
log_accuracy = accuracy_score(Y_test, y_pred_log)
log_precision = precision_score(Y_test, y_pred_log, average='weighted')
log_recall = recall_score(Y_test, y_pred_log, average='weighted')
log_f1 = f1_score(Y_test, y_pred_log, average='weighted')

# Prepare data for Random Forest
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels_encoded, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Collect Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

# Prepare features and labels for TensorFlow and TensorFlow Lite
labels_one_hot = pd.get_dummies(data['label'])
features_normalized_tf = scaler.fit_transform(features)
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(features_normalized_tf, labels_one_hot, test_size=0.2, random_state=42)

# TensorFlow Model
tf_model = Sequential([
    Dense(256, activation='relu', input_dim=7),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(labels_one_hot.shape[1], activation='softmax')
])

tf_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tf_model.fit(X_train_tf, y_train_tf, epochs=100, batch_size=64, validation_data=(X_test_tf, y_test_tf), callbacks=[early_stopping])

# Evaluate TensorFlow model
test_loss_tf, test_acc_tf = tf_model.evaluate(X_test_tf, y_test_tf)
y_pred_tf = tf_model.predict(X_test_tf).argmax(axis=1)
y_true_tf = y_test_tf.to_numpy().argmax(axis=1)

# Collect TensorFlow metrics
tf_precision = precision_score(y_true_tf, y_pred_tf, average='weighted')
tf_recall = recall_score(y_true_tf, y_pred_tf, average='weighted')
tf_f1 = f1_score(y_true_tf, y_pred_tf, average='weighted')

# Convert the TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Function to make predictions with the TFLite model
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = input_data.reshape(1, -1).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Evaluate the TFLite model on the test data
predictions_tflite = []
for i in range(len(X_test_tf)):
    test_sample = X_test_tf[i].reshape(1, -1).astype(np.float32)
    prediction = predict_tflite(interpreter, test_sample)
    predictions_tflite.append(prediction.argmax(axis=1))

y_pred_tflite = np.array(predictions_tflite).flatten()

# Collect TensorFlow Lite metrics
tflite_accuracy = accuracy_score(y_true_tf, y_pred_tflite)
tflite_precision = precision_score(y_true_tf, y_pred_tflite, average='weighted')
tflite_recall = recall_score(y_true_tf, y_pred_tflite, average='weighted')
tflite_f1 = f1_score(y_true_tf, y_pred_tflite, average='weighted')

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "TensorFlow", "TensorFlow Lite"],
    "Accuracy": [log_accuracy, rf_accuracy, test_acc_tf, tflite_accuracy],
    "Precision": [log_precision, rf_precision, tf_precision, tflite_precision],
    "Recall": [log_recall, rf_recall, tf_recall, tflite_recall],
    "F1-Score": [log_f1, rf_f1, tf_f1, tflite_f1]
})

# Display the metrics table
print("\nModel Comparison Table:")
print(metrics_df)