
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow:", tf.__version__)

print("\n--- Loading 'tf_flowers' ---")
(train_ds, validation_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
class_names = metadata.features['label'].names
print(f"Found {num_classes} classes: {class_names}")

IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(format_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

validation_ds = validation_ds.map(format_image, num_parallel_calls=AUTOTUNE)
validation_ds = validation_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds.map(format_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Data pipeline ready.")

print("\n--- Building model with ResNet50 as base ---")
base_model = tf.keras.applications.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Corrected typo here
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

print("Model summary:")
model.summary()

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

print("\n--- Stage 2: Fine-tuning top layers of base model ---")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Unfroze layers from {fine_tune_at} to {len(base_model.layers)-1}")

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

print("\n--- Evaluating on test set ---")
loss, acc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4%}")

print("\n--- Preparing Grad-CAM ---")

resnet_layer_in_model = None
for layer in model.layers:
    if layer.name == 'resnet50':
        resnet_layer_in_model = layer
        break

if resnet_layer_in_model is None:
    raise RuntimeError("Could not find the 'resnet50' layer in the main model.")

last_conv_layer_in_resnet = None
for layer in reversed(resnet_layer_in_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
        last_conv_layer_in_resnet = layer
        break

if last_conv_layer_in_resnet is None:
    raise RuntimeError("Could not find a Conv2D layer inside the ResNet50 base model.")

target_conv_layer_name = last_conv_layer_in_resnet.name
print("Last conv layer in base_model (chosen for Grad-CAM):", target_conv_layer_name)

grad_cam_intermediate_feature_extractor = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=resnet_layer_in_model.get_layer(target_conv_layer_name).output
)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[grad_cam_intermediate_feature_extractor.output, model.output]
)
print("Built grad_model mapping inputs -> (conv_outputs, predictions).")

def get_gradcam_heatmap(img_array, grad_model_obj, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model_obj(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("Warning: gradients are None. Returning zero heatmap with conv spatial shape.")
        conv_shape = conv_outputs.shape
        if len(conv_shape) >= 3:
            h, w = int(conv_shape[1]), int(conv_shape[2])
            return np.zeros((h, w), dtype=np.float32)
        else:
            return np.zeros((7, 7), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    heatmap_np = heatmap.numpy()
    return heatmap_np

def display_gradcam(original_img, heatmap, alpha=0.6):
    if heatmap.ndim == 2:
        heatmap = heatmap[..., np.newaxis]
    elif heatmap.ndim == 3 and heatmap.shape[-1] != 1:
        heatmap = np.mean(heatmap, axis=-1, keepdims=True)

    heatmap_resized = tf.image.resize(heatmap, (original_img.shape[0], original_img.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-10)

    cmap = plt.cm.get_cmap("jet")
    cmap_colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = cmap_colors[np.uint8(255 * heatmap_resized)]

    orig_uint8 = np.uint8(np.clip(original_img * 255.0, 0, 255))
    jet_heatmap_uint8 = np.uint8(np.clip(jet_heatmap * 255.0, 0, 255))
    superimposed = np.uint8(np.clip(jet_heatmap_uint8 * alpha + orig_uint8 * (1 - alpha), 0, 255))

    plt.imshow(superimposed)
    plt.axis('off')

print("\n--- Generating Grad-CAM for one test image ---")
image_batch, label_batch = next(iter(test_ds))
image_to_analyze = image_batch[0].numpy()
true_label_idx = int(label_batch[0].numpy())
true_label = class_names[true_label_idx]

input_for_model = np.expand_dims(image_to_analyze, axis=0)
preds = model.predict(input_for_model)
pred_idx = int(np.argmax(preds[0]))
pred_label = class_names[pred_idx]
print(f"True label: {true_label}  |  Predicted: {pred_label}")

heatmap = get_gradcam_heatmap(input_for_model, grad_model, pred_index=pred_idx)
print("Heatmap shape (conv-space):", heatmap.shape)

if np.allclose(heatmap, 0):
    print("Note: heatmap is all zeros (no positive activations). Overlay will be blank.")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_to_analyze)
plt.title(f"Original\nTrue: {true_label}\nPred: {pred_label}")
plt.axis('off')

plt.subplot(1, 2, 2)
display_gradcam(image_to_analyze, heatmap, alpha=0.5)
plt.title("Grad-CAM")
plt.axis('off')
plt.tight_layout()
plt.show()

print("Done.")
