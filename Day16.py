# --- 1. Import Necessary Libraries ---
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")


# --- 2. Load and Prepare the Fashion-MNIST Dataset ---
print("\n--- Loading and Preparing Data ---")
# Load the dataset directly from Keras
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values from the [0, 255] range to the [0, 1] range.
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split the full training set into a smaller training set and a validation set.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Define the human-readable class names for later plotting
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Data loaded and split successfully.")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_valid.shape}")
print(f"Test set shape: {X_test.shape}")


# --- 3. Build the Neural Network with Regularization ---
print("\n--- Building the Neural Network ---")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

print("Model built successfully.")
model.summary()


# --- 4. Define Advanced Training Callbacks ---
print("\n--- Defining Training Callbacks ---")
# EarlyStopping callback to stop training when performance on the validation set stops improving.
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=5,
    monitor='val_loss',
    restore_best_weights=True
)

# THE FIX: This function now explicitly converts its output to a standard Python float.
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        # Calculate the new learning rate
        new_lr = lr * tf.math.exp(-0.1)
        # Convert the TensorFlow tensor to a float before returning
        return float(new_lr)

# Create the LearningRateScheduler callback with our corrected function
lr_scheduler_cb = keras.callbacks.LearningRateScheduler(scheduler)

# A list of all callbacks to be used during training
callbacks = [early_stopping_cb, lr_scheduler_cb]
print("Callbacks defined: EarlyStopping and LearningRateScheduler.")


# --- 5. Compile and Train the Model ---
print("\n--- Compiling and Training the Model ---")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

# Train the model with the specified callbacks.
# EarlyStopping will likely stop the training before 100 epochs.
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=callbacks,
                    verbose=2)

print("\nModel training complete.")


# --- 6. Evaluate the Final Model on the Test Set ---
print("\n--- Evaluating Final Model Performance ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy:.2%})")

# Generate predictions to create detailed reports
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Print a detailed classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualize the confusion matrix
print("\n--- Generating Confusion Matrix ---")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Fashion-MNIST Classification', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()