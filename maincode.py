import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MNIST data set loader
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Pixel value range between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 images into 784-element vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # First hidden layer with 128 neurons
    layers.Dense(64, activation='relu'),                        # Second hidden layer with 64 neurons
    layers.Dense(10, activation='softmax')                      # Output layer for 10 digit classes
])

# Model compiling 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model 
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Predictor
predictions = model.predict(x_test)

# Get the predicted value
predicted_label = np.argmax(predictions[0])
print(f"\nPredicted label for the first test image: {predicted_label}")

# To show the image
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
