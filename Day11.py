# --- 1. Import Libraries ---
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print("All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")


# --- 2. Define Dataset Paths (MANUAL SETUP) ---
# IMPORTANT: Manually download and unzip the dataset into your project folder first.
# Download from: https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

# --- THE FIX: SIMPLIFIED THE PATH ---
# The script will now look for the 'train' folder directly inside 'cats_and_dogs_filtered'
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# --- END OF FIX ---

# Check if the path exists to give a helpful error message
if not os.path.exists(train_dir):
    print("="*50)
    print("ERROR: The training directory was not found at the specified path.")
    print(f"Please check your folder structure. The script is looking for: '{train_dir}'")
    print("="*50)
    exit() # Stop the script if data is not found
else:
    print("\nDataset folder found successfully.")


# --- 3. Create Data Generators ---
print("\n--- Creating Data Generators ---")
# Use separate generators for train and validation since the data is already split
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
print("Data generators created.")


# --- 4. Build the Transfer Learning Model ---
print("\n--- Building the Transfer Learning Model ---")
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
print("Model built successfully.")
model.summary()


# --- 5. Compile and Train the Model ---
print("\n--- Compiling and Training the Model ---")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)
print("Model training complete.")


# --- 6. Make and Visualize Predictions ---
print("\n--- Making and Visualizing Predictions ---")

test_images, test_labels = next(validation_generator)
predictions_prob = model.predict(test_images)
class_names = list(train_generator.class_indices.keys())

def display_predictions(images, true_labels, preds_prob, class_names):
    plt.figure(figsize=(15, 15))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        true_label_name = class_names[int(true_labels[i])]
        
        confidence = preds_prob[i][0]
        predicted_class_index = 1 if confidence > 0.5 else 0
        predicted_class_name = class_names[predicted_class_index]
        
        if predicted_class_index == 0:
            confidence = 1 - confidence
            
        title_color = 'green' if predicted_class_name == true_label_name else 'red'
        
        plt.title(f"True: {true_label_name}\nPred: {predicted_class_name} ({confidence:.2%})", color=title_color, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_predictions(test_images, test_labels, predictions_prob, class_names)