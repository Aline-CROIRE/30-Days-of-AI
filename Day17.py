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


# --- 2. Load and Prepare the CIFAR-10 Dataset ---
print("\n--- Loading and Preparing CIFAR-10 Data ---")
# Load the dataset directly from Keras. It consists of 60,000 32x32 color images in 10 classes.
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values from the [0, 255] range to the [0, 1] range.
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split the full training set into a smaller training set and a validation set.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Define the human-readable class names for later plotting
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

print(f"Data loaded and split successfully.")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_valid.shape}")
print(f"Test set shape: {X_test.shape}")


# --- 3. Set up Data Augmentation ---
print("\n--- Setting up Data Augmentation Generator ---")
# Create an ImageDataGenerator to apply random transformations to the training images.
# This helps the model generalize better and prevents overfitting.
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Fit the generator on the training data.
datagen.fit(X_train)
print("Data augmentation generator is ready.")


# --- 4. Build the Convolutional Neural Network (CNN) ---
print("\n--- Building the CNN ---")
# Using the Sequential API for a linear stack of layers
model = keras.models.Sequential([
    # Input shape is 32x32 pixels with 3 color channels (RGB)
    
    # -- First Convolutional Block --
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=[32, 32, 3]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    # -- Second Convolutional Block --
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    # -- Third Convolutional Block --
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    # -- Flatten and Dense Layers for Classification --
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax') # 10 output neurons for 10 classes
])

print("Model built successfully.")
model.summary()


# --- 5. Compile and Train the Model ---
print("\n--- Compiling and Training the Model (this will take some time)... ---")
# Compile the model with an optimizer, loss function, and metrics
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

# Set up Early Stopping to prevent overfitting and save time
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model using the data generator for augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=100, # Set a high number, EarlyStopping will finish sooner
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping_cb],
                    verbose=2)

print("\nModel training complete.")


# --- 6. Evaluate the Final Model on the Test Set ---
print("\n--- Evaluating Final Model Performance ---")
# Evaluate the model on the unseen test data
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
plt.title('Confusion Matrix for CIFAR-10 Classification', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()