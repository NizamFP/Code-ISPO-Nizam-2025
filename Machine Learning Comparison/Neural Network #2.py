# This code uses Neural Network Model #2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Load the data
PATH = 'Crop_recommendation.csv'
data = pd.read_csv(PATH)

# Prepare features and labels
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']

# Encode labels if they are categorical
labels = pd.get_dummies(labels)

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a neural network model with similar complexity to a Random Forest
model = Sequential([
    Dense(256, activation='relu', input_dim=7),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('crop_recommendation_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Convert y_test DataFrame to a NumPy array
y_test_np = y_test.to_numpy()

# Function to make predictions with the TFLite model
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()

    # Reshape input data to match the model's input shape
    input_data = input_data.reshape(1, -1).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Evaluate the TFLite model on the test data
predictions = []
for i in range(len(X_test)):
    test_sample = X_test[i].reshape(1, -1).astype(np.float32)
    prediction = predict_tflite(interpreter, test_sample)
    predictions.append(prediction.argmax(axis=1))

# Calculate accuracy for the TFLite model
test_acc_tflite = np.mean(y_test_np.argmax(axis=1) == np.array(predictions).flatten())
print('Test accuracy (TFLite):', test_acc_tflite)