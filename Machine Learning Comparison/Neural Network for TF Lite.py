# This code uses Neural Network Model for Quantized TF Lite

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Create a neural network model with enhanced architecture
model = Sequential([
    Input(shape=(7,)),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(labels.shape[1], activation='softmax')
])

# Compile the model with an adjusted optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Train the model with early stopping and learning rate reduction
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

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

# Quantize the model for ESP32 compatibility
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization to reduce the model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide a representative dataset for quantization
def representative_dataset_generator():
    for value in X_train[:100]:  # Use a small sample of your training data
        yield [value.astype(np.float32)]

converter.representative_dataset = representative_dataset_generator

# Specify that we want to use integer quantization for compatibility with microcontrollers
converter.target_spec.supported_types = [tf.int8]

# Convert the model to a quantized TFLite model
tflite_quantized_model = converter.convert()

# Save the quantized TFLite model
with open('crop_recommendation_model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)

print("Quantized model saved as crop_recommendation_model_quantized.tflite")

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_path='crop_recommendation_model_quantized.tflite')
interpreter.allocate_tensors()

# Function to make predictions with the quantized TFLite model
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Reshape input data to match the model's input shape
    input_data = input_data.reshape(1, -1).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Evaluate the quantized TFLite model on the test data
predictions = []
for i in range(len(X_test)):
    test_sample = X_test[i].reshape(1, -1).astype(np.float32)
    prediction = predict_tflite(interpreter, test_sample)
    predictions.append(prediction.argmax(axis=1))

# Calculate accuracy for the quantized TFLite model
test_acc_tflite_quantized = np.mean(y_test_np.argmax(axis=1) == np.array(predictions).flatten())
print('Test accuracy (Quantized TFLite):', test_acc_tflite_quantized)