# full_corrected_flowers_resnet50_gradcam.py
# --- 0. Notes ---
# - Restart your kernel if you recently changed TensorFlow / Keras installations.
# - This script expects tensorflow, tensorflow_datasets, numpy, matplotlib to be installed.
# - If you encounter environment issues (e.g., Keras import errors, RecursionError):
#   1. Run 'pip uninstall keras' (if you installed it separately).
#   2. Run 'pip install tf-keras'.
#   3. IMPORTANT: Restart your Python kernel or environment completely.
#   4. If issues with tfds persist: 'pip uninstall tensorflow-datasets; pip install tensorflow-datasets; restart.'

# --- 1. Imports & Setup ---
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow:", tf.__version__)

# --- 2. Load and Prepare the Flowers Dataset ---
print("\n--- Loading 'tf_flowers' ---")
try:
    (train_ds, validation_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
except Exception as e:
    raise RuntimeError("Failed loading 'tf_flowers' with tfds. "
                       "Check tensorflow-datasets installation and restart your kernel.") from e

num_classes = metadata.features['label'].num_classes
class_names = metadata.features['label'].names
print(f"Found {num_classes} classes: {class_names}")

# --- 3. Data pipeline ---
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def format_image(image, label):
    # image from tfds is uint8 [0,255]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0  # normalized to [0,1]
    return image, label

train_ds = train_ds.map(format_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

validation_ds = validation_ds.map(format_image, num_parallel_calls=AUTOTUNE)
validation_ds = validation_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds.map(format_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Data pipeline ready.")

# --- 4. Build the Transfer Learning Model (ResNet50) ---
print("\n--- Building model with ResNet50 as base ---")
try:
    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
except Exception as e:
    raise RuntimeError(
        "Failed to load ResNet50. This often means a Keras/TensorFlow environment conflict. "
        "If you installed 'keras' separately try: pip uninstall keras; pip install tf-keras; then restart your kernel."
    ) from e

# Freeze base for initial training
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

print("Model summary:")
model.summary()

# --- 5. Stage 1: Initial training (head only) ---
print("\n--- Stage 1: Training classification head ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

initial_epochs = 5
history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=validation_ds
)

# --- 6. Stage 2: Fine-tuning ---
print("\n--- Stage 2: Fine-tuning top layers of base model ---")
# Unfreeze the top N layers of the base model
base_model.trainable = True
# Fine-tune only the last 20 layers (you can change this)
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Unfroze layers from {fine_tune_at} to {len(base_model.layers)-1} (top {len(base_model.layers)-fine_tune_at} layers)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_ds
)

print("Fine-tuning finished.")

# --- 7. Evaluation ---
print("\n--- Evaluating on test set ---")
loss, acc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4%}")

# --- 8. Grad-CAM Implementation (robust) ---
print("\n--- Preparing Grad-CAM ---")

# Use the `base_model` directly (we created it earlier)
# Find last convolutional layer inside base_model
last_conv_layer = None
for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise RuntimeError("Could not find a Conv2D layer inside the base model.")

print("Last conv layer:", last_conv_layer.name)

# Build a model that maps the base model input to the last conv layer output
grad_model = tf.keras.models.Model(inputs=base_model.input, outputs=last_conv_layer.output)

# Helper: compute Grad-CAM heatmap
def get_gradcam_heatmap(img_array, grad_model, full_model, pred_index=None):
    """
    img_array: shape (1, H, W, 3), float32, values [0,1]
    grad_model: maps input -> last conv outputs
    full_model: the full classification model
    """
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        # Forward pass through grad_model to get conv outputs
        last_conv = grad_model(img_tensor)
        tape.watch(last_conv)
        preds = full_model(img_tensor, training=False)  # shape (1, num_classes)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the class output w.r.t. last conv feature map
    grads = tape.gradient(class_channel, last_conv)  # shape (1, Hc, Wc, C)
    # Global average pool the gradients over (Hc, Wc)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # shape (1, C)
    pooled_grads = pooled_grads[0]  # shape (C,)

    last_conv_output = last_conv[0]  # shape (Hc, Wc, C)
    # Multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
    heatmap = tf.reduce_sum(last_conv_output * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)
    return heatmap.numpy()  # 2D array shape (Hc, Wc)

# Helper: display Grad-CAM over image
def display_gradcam(original_img, heatmap, alpha=0.6):
    """
    original_img: HxW x3, float32 in [0,1]
    heatmap: 2D numpy array with values in [0,1]
    """
    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (original_img.shape[0], original_img.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    # Convert heatmap to RGB using a colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    cmap = plt.cm.get_cmap("jet")
    cmap_colors = cmap(np.arange(256))[:, :3]  # shape (256,3) floats in [0,1]
    jet_heatmap = cmap_colors[heatmap_uint8]  # shape (H, W, 3) floats in [0,1]
    jet_heatmap = jet_heatmap * 255.0  # scale to [0,255]

    # original_img is [0,1] -> scale to [0,255]
    orig_uint8 = np.uint8(np.clip(original_img * 255.0, 0, 255))

    superimposed = np.uint8(np.clip(jet_heatmap * alpha + orig_uint8 * (1 - alpha), 0, 255))

    plt.imshow(superimposed)
    plt.axis('off')

# --- 9. Pick one image from the test set and visualize Grad-CAM ---
print("\n--- Generating Grad-CAM for one test image ---")
# Get one batch from test_ds
image_batch, label_batch = next(iter(test_ds))
image_to_analyze = image_batch[0].numpy()  # shape (H,W,3), values [0,1]
true_label_idx = int(label_batch[0].numpy())
true_label = class_names[true_label_idx]

# Predict
preds = model.predict(np.expand_dims(image_to_analyze, axis=0))
pred_idx = int(np.argmax(preds[0]))
pred_label = class_names[pred_idx]

# Compute heatmap
heatmap = get_gradcam_heatmap(np.expand_dims(image_to_analyze, axis=0), grad_model, model, pred_index=pred_idx)

# Plot original and grad-cam
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image_to_analyze)
# --- CHANGE HERE: Explicitly label as "Original Image" ---
plt.title(f"Original Image\nTrue: {true_label}\nPred: {pred_label}")
plt.axis('off')

plt.subplot(1,2,2)
display_gradcam(image_to_analyze, heatmap, alpha=0.5)
plt.title("Grad-CAM")
plt.axis('off')
plt.tight_layout()
plt.show()

print("Done.")