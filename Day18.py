# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import requests

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")


# --- 2. Download and Load the Dataset ---
data_path = 'airline-passengers.csv'
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'

if not os.path.exists(data_path):
    print(f"Dataset not found. Downloading...")
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        with open(data_path, 'w') as f:
            f.write(response.text)
        print("Download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        exit()

df = pd.read_csv(data_path, index_col='Month', parse_dates=True)
data = df['Passengers'].values.astype(float)
print(f"\nDataset loaded successfully.")


# --- 3. Preprocess the Data (Scaling and Windowing) ---
print("\n--- Preprocessing Data ---")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 12
X, y = create_dataset(data_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.67)
X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]
y_train, y_test = y[0:train_size], y[train_size:len(y)]
print("Data prepared.")


# --- 4. Build the LSTM Model with Attention (Functional API) ---
print("\n--- Building the LSTM Model with Attention ---")
inputs = keras.layers.Input(shape=(look_back, 1))
lstm_out = keras.layers.LSTM(50, return_sequences=True)(inputs)
attention_out = keras.layers.Attention()([lstm_out, lstm_out])
flattened_attention = keras.layers.Flatten()(attention_out)
dense1 = keras.layers.Dense(25, activation='relu')(flattened_attention)
outputs = keras.layers.Dense(1)(dense1)
model = keras.Model(inputs=inputs, outputs=outputs)
print("Model built successfully.")
model.summary()


# --- 5. Compile and Train the Model ---
print("\n--- Compiling and Training the Model... ---")
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping_cb])
print("\nModel training complete.")


# --- 6. Evaluate and Visualize the Forecast (DEFINITIVELY CORRECTED PLOTTING) ---
print("\n--- Evaluating Model and Generating Forecast Plot ---")
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict_inv = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict_inv = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

train_rmse = np.sqrt(mean_squared_error(y_train_inv[0], train_predict_inv[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict_inv[:, 0]))
print(f"Train RMSE: {train_rmse:.2f} Passengers")
print(f"Test RMSE: {test_rmse:.2f} Passengers")

# --- THE FIX: The Simplest and Most Foolproof Plotting Method ---
# Create an empty array for train predictions
train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
# Place train predictions at the correct positions
train_predict_plot[look_back:len(train_predict_inv) + look_back, :] = train_predict_inv

# Create an empty array for test predictions
test_predict_plot = np.empty_like(data_scaled)
test_predict_plot[:, :] = np.nan
# Calculate the starting position for test predictions
test_start_point = len(data_scaled) - len(test_predict_inv) - 1
# Place test predictions at the correct positions
test_predict_plot[test_start_point:len(data_scaled) - 1, :] = test_predict_inv

# Create the plot
plt.figure(figsize=(15, 7))
plt.plot(scaler.inverse_transform(data_scaled), label='Original Data', color='blue')
plt.plot(train_predict_plot, label='Training Predictions', color='orange')
plt.plot(test_predict_plot, label='Test Forecast', color='green')
# --- END OF FIX ---

plt.title('Airline Passenger Forecasting with LSTM & Attention', fontsize=16)
plt.xlabel('Time (Months since start)', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()