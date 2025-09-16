# --- 1. Import Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print("All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")


# --- 2. Generate and Prepare the Dataset ---
print("\n--- Generating Synthetic 'Moons' Dataset ---")
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1) # Reshape y to be a column vector for matrix operations

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualize the training data
plt.figure(figsize=(8, 5))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()


# ===================================================================
# PART 1: NEURAL NETWORK FROM SCRATCH WITH NUMPY
# ===================================================================

print("\n" + "="*50)
print("PART 1: Building a Neural Network from Scratch with NumPy")
print("="*50)

# --- 3. Define NumPy Network Components ---
# Activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Note: We pass the ACTIVATED output to the derivative, not the raw input
    return x * (1 - x)

# Network architecture
input_neurons = X_train.shape[1]
hidden_neurons = 4
output_neurons = 1

# Hyperparameters
learning_rate = 0.01
epochs = 10000

# Initialize weights and biases with a seed for reproducibility
np.random.seed(42)
weights_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
weights_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))


# --- 4. The NumPy Training Loop ---
print("\n--- Starting NumPy Model Training ---")
loss_history_numpy = []
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X_train, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate Loss (Mean Squared Error for simplicity in this part)
    error = y_train - predicted_output
    loss = np.mean(error**2)
    loss_history_numpy.append(loss)
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update Weights and Biases
    weights_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("NumPy model training complete.")


# --- 5. Visualize NumPy Model Learning ---
plt.figure(figsize=(10, 6))
plt.plot(loss_history_numpy)
plt.title("Loss Curve for NumPy Model")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.grid(True)
plt.show()


# ===================================================================
# PART 2: THE SAME NEURAL NETWORK WITH TENSORFLOW/KERAS
# ===================================================================

print("\n" + "="*50)
print("PART 2: Building the same Neural Network with Keras")
print("="*50)

# --- 6. Define the Keras Model Architecture ---
# The Sequential model is a linear stack of layers, perfect for our MLP.
model = keras.Sequential([
    # A Dense layer is a fully connected layer.
    # We specify the number of neurons, activation function, and input shape for the first layer.
    keras.layers.Dense(hidden_neurons, input_shape=(input_neurons,), activation='sigmoid'),
    # For subsequent layers, Keras infers the input shape.
    keras.layers.Dense(output_neurons, activation='sigmoid')
])

# --- 7. Compile the Keras Model ---
# This step configures the model for training.
# Optimizer: Algorithm to update the weights (Adam is a sophisticated, popular choice).
# Loss: The function to measure the error (binary_crossentropy is standard for binary classification).
# Metrics: What to display during training.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- Keras Model Summary ---")
model.summary()


# --- 8. Train the Keras Model ---
print("\n--- Starting Keras Model Training ---")
# The .fit() method handles the entire training loop for us.
# verbose=2 will print a summary line for each epoch.
history = model.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    validation_data=(X_test, y_test),
    verbose=0 # Set to 1 or 2 to see epoch-by-epoch progress
)
print("Keras model training complete.")


# --- 9. Evaluate and Visualize Keras Model Learning ---
# .evaluate() computes the loss and metrics on the unseen test data.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n--- Keras Model Performance on Test Set ---")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

# Plot the Keras loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curves for Keras Model")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()