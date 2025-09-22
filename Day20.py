import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- Loading and Preparing MNIST Data ---")
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
print(f"Data loaded. x_train shape: {x_train.shape}")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
decoder_outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

print("\n--- Training the VAE (this will take a few minutes)... ---")
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128, verbose=2)
print("VAE training complete.")


print("\n--- Generating Visualization of Original, Reconstructed, and AI-Generated Digits ---")

n_to_show = 15
_, _, z_sample_reconstruct = vae.encoder.predict(x_test[:n_to_show], verbose=0)
reconstructions = vae.decoder.predict(z_sample_reconstruct, verbose=0)

n_to_generate_grid = 15
grid_x = np.linspace(-2.5, 2.5, n_to_generate_grid)
grid_y = np.linspace(-2.5, 2.5, n_to_generate_grid)[::-1]

generated_digits_grid = np.zeros((28 * n_to_generate_grid, 28 * n_to_generate_grid))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample_generate = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample_generate, verbose=0)
        digit = x_decoded[0].reshape(28, 28)
        generated_digits_grid[
            i * 28 : (i + 1) * 28,
            j * 28 : (j + 1) * 28,
        ] = digit

fig, axes = plt.subplots(3, 1, figsize=(15, 22), gridspec_kw={'height_ratios': [1, 1, 15]})
fig.suptitle("VAE: Reconstruction and Generation Showcase", fontsize=20)

# --- Subplot for Original Digits ---
original_strip = np.hstack([x_test[i].reshape(28, 28) for i in range(n_to_show)])
axes[0].imshow(original_strip, cmap='gray', aspect='auto')
axes[0].set_title("Original Digits from Test Set", fontsize=14)
axes[0].axis('off')

# --- Subplot for Reconstructed Digits ---
reconstructed_strip = np.hstack([reconstructions[i].reshape(28, 28) for i in range(n_to_show)])
axes[1].imshow(reconstructed_strip, cmap='gray', aspect='auto')
axes[1].set_title("VAE's Reconstructed Versions", fontsize=14)
axes[1].axis('off')

# --- Subplot for Generated Digits ---
axes[2].imshow(generated_digits_grid, cmap='gray')
axes[2].set_title("AI-Generated Digits 'Dreamed' from the Latent Space", fontsize=14)
axes[2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()