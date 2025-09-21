# --- 1. Import Libraries ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- 2. Load and Prepare the MNIST Dataset ---
print("\n--- Loading and Preparing MNIST Data ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1 and flatten the images
# The VAE will work with flat vectors of 784 pixels.
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)

print(f"Data loaded. x_train shape: {x_train.shape}")


# --- 3. Build the VAE Components ---

# --- 3a. The Sampling Layer (Reparameterization Trick) ---
# This custom layer is the core of the VAE. It takes the mean and log-variance
# from the encoder and uses them to sample a point from the latent distribution.
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # Generate a random vector from a standard normal distribution
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # Combine them to sample from the learned distribution
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- 3b. The Encoder ---
# This network maps the input images to the parameters of the latent distribution.
latent_dim = 2 # The latent space will be 2D for easy visualization

encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
# The encoder outputs two vectors: one for the means, one for the log-variances.
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# The sampling layer creates the final latent vector 'z'.
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print("\nEncoder built successfully.")
encoder.summary()

# --- 3c. The Decoder ---
# This network maps points from the latent space back to the original image space.
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
# The output layer has 784 neurons (one for each pixel) and a sigmoid activation
# to ensure pixel values are between 0 and 1.
decoder_outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
print("\nDecoder built successfully.")
decoder.summary()


# --- 4. Define the Full VAE Model with Custom Training Logic ---
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Define trackers for our custom losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # --- Calculate the two-part loss ---
            # 1. Reconstruction Loss (how well we reconstructed the image)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
            )
            # 2. KL Divergence Loss (a regularization term that keeps the latent space smooth)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # The final loss is the sum of the two
            total_loss = reconstruction_loss + kl_loss

        # Standard backpropagation and weight updates
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update our metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# --- 5. Train the VAE ---
print("\n--- Training the VAE (this will take a few minutes)... ---")
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128, verbose=2)
print("VAE training complete.")


# --- 6. Use the VAE for Reconstruction, Anomaly Detection, and Generation ---

# --- Task 1: Reconstruct Digits ---
print("\n--- Task 1: Reconstructing Digits ---")
n_to_show = 10
# We use the encoder's third output (the sampled 'z') to feed into the decoder.
z_mean, _, z_sample = vae.encoder.predict(x_test[:n_to_show])
reconstructions = vae.decoder.predict(z_sample)

fig, axes = plt.subplots(2, n_to_show, figsize=(20, 4))
for i in range(n_to_show):
    axes[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")
plt.suptitle("Task 1: Original vs. Reconstructed Digits", fontsize=16)
plt.show()

# --- Task 2: Anomaly Detection ---
print("\n--- Task 2: Detecting Anomalies ---")
# Load the Fashion-MNIST dataset to use as our "anomalies"
(x_fashion, _), (_, _) = keras.datasets.fashion_mnist.load_data()
x_fashion = x_fashion.astype("float32") / 255.0
x_fashion = x_fashion.reshape(x_fashion.shape[0], -1)

# A function to calculate reconstruction error
def get_reconstruction_error(data):
    _, _, z = vae.encoder.predict(data, verbose=0)
    reconstruction = vae.decoder.predict(z, verbose=0)
    # Use binary_crossentropy to calculate the pixel-wise error
    errors = tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
    return errors

# Calculate error for normal and anomaly data
error_normal = get_reconstruction_error(x_test)
error_anomaly = get_reconstruction_error(x_fashion)

plt.figure(figsize=(10, 6))
plt.hist(error_normal, bins=50, density=True, label='Normal (Digits)', alpha=0.7)
plt.hist(error_anomaly, bins=50, density=True, label='Anomaly (Clothes)', alpha=0.7)
plt.title("Task 2: Reconstruction Error for Anomaly Detection")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.legend()
plt.show()
print("Note how the anomaly data (clothes) has a much higher reconstruction error.")

# --- Task 3: Generate New Digits ---
print("\n--- Task 3: Generating New Digits ---")
n_to_generate = 15
# We will sample points from a 2D grid in the latent space
grid_x = np.linspace(-2, 2, n_to_generate)
grid_y = np.linspace(-2, 2, n_to_generate)[::-1]

fig, axes = plt.subplots(n_to_generate, n_to_generate, figsize=(12, 12))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        # Create a 2D latent vector for each point in the grid
        z_sample = np.array([[xi, yi]])
        # Decode the latent vector back into an image
        x_decoded = vae.decoder.predict(z_sample, verbose=0)
        digit = x_decoded[0].reshape(28, 28)
        axes[i, j].imshow(digit, cmap="gray")
        axes[i, j].axis("off")

plt.suptitle("Task 3: New Digits Generated from the Latent Space", fontsize=16)
plt.show()