# --- 1. Import Necessary Libraries ---
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# --- 2. Load and Prepare the Flowers Dataset ---
print("\n--- Loading and Preparing the 'tf_flowers' Dataset ---")
(train_ds, validation_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
class_names = metadata.features['label'].names
print(f"Dataset loaded. Found {num_classes} classes: {class_names}")


# --- 3. Create an Efficient Data Pipeline ---
IMG_SIZE = 224
def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
train_ds = train_ds.map(format_image).cache().shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.map(format_image).cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(format_image).cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
print("Data pipeline created.")


# --- 4. Build the Transfer Learning Model ---
print("\n--- Building the Transfer Learning Model using ResNet50 ---")
base_model = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                            include_top=False,
                                            weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print("Model built successfully.")
model.summary()


# --- 5. Stage 1: Initial Training ---
print("\n--- Stage 1: Initial Training (Training the new classification head) ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds,
                    epochs=5,
                    validation_data=validation_ds)


# --- 6. Stage 2: Fine-Tuning ---
print("\n--- Stage 2: Fine-Tuning (Unfreezing and training top layers of the base model) ---")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
print(f"Unfroze the top {len(base_model.layers) - fine_tune_at} layers of the base model.")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs = 5 + fine_tune_epochs
history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_ds)
print("\nFine-tuning complete.")


# --- 7. Final Evaluation and Grad-CAM Visualization (Corrected) ---
print("\n--- Final Evaluation on Test Set ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy after fine-tuning: {accuracy:.2%}")

# --- Grad-CAM Implementation ---
print("\n--- Generating Grad-CAM Visualization ---")

# --- THE FIX: Create a simpler Grad-CAM model focused on the base model ---
# 1. Get the ResNet50 base model layer. Note that Keras renames it in the full model.
base_model_from_main_model = model.get_layer('resnet50')

# 2. Find the name of the last convolutional layer *within the base model*.
last_conv_layer_name = [layer.name for layer in base_model_from_main_model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
last_conv_layer = base_model_from_main_model.get_layer(last_conv_layer_name)

# 3. Create a new model that maps the base model's input to the last conv layer's output.
grad_model = tf.keras.models.Model(
    [base_model_from_main_model.inputs], [last_conv_layer.output]
)
# --- END OF FIX ---

# Helper function to compute the Grad-CAM heatmap
def get_gradcam_heatmap(img_array, grad_model, full_model, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output = grad_model(img_array)
        tape.watch(last_conv_layer_output) # Watch the conv layer output
        preds = full_model(img_array) # Get predictions from the full model
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# Helper function to display the heatmap
def display_gradcam(img, heatmap, alpha=0.6):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img * 255
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    plt.imshow(superimposed_img)

# Visualization logic
image_batch, label_batch = next(iter(test_ds))
image_to_analyze = image_batch[0]
img_array = np.expand_dims(image_to_analyze, axis=0)
heatmap = get_gradcam_heatmap(img_array, grad_model, model) # Pass the full model to the function now
true_label = class_names[label_batch[0]]
preds = model.predict(img_array)
pred_label = class_names[np.argmax(preds)]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_to_analyze)
plt.title(f"Original Image\nTrue Label: {true_label}\nPredicted Label: {pred_label}")
plt.axis('off')
plt.subplot(1, 2, 2)
display_gradcam(image_to_analyze, heatmap)
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.tight_layout()
plt.show()