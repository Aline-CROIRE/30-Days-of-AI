# --- 1. SETUP: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# --- 2. DATA LOADING ---
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

print(f"\nDataset loaded. X shape: {X.shape}")


# --- 3. FEATURE ENGINEERING: Create Polynomial Features ---
print("\nGenerating Polynomial Features (degree=2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Number of features transformed from {X.shape[1]} to {X_poly.shape[1]}")


# --- 4. DATA PREPARATION: Split and Scale ---
print("\nSplitting and scaling the data...")
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")


# --- 5. MODEL 1: Linear Regression (Baseline) ---
print("\n--- Training Baseline Model: Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# Evaluate Linear Regression
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("--- Linear Regression Performance ---")
print(f"Root Mean Squared Error (RMSE): {rmse_lin:.4f}")
print(f"R-squared (R²): {r2_lin:.4f}")


# --- 6. MODEL 2: Ridge Regression (Regularized) ---
print("\n--- Training Regularized Model: Ridge Regression ---")
ridge_reg = Ridge(alpha=10) # Using a slightly higher alpha for a clearer difference
ridge_reg.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_scaled)

# Evaluate Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("--- Ridge Regression Performance ---")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge:.4f}")
print(f"R-squared (R²): {r2_ridge:.4f}")


# --- 7. VISUALIZATION: Actual vs. Predicted Values ---
print("\n--- Generating Visualizations ---")

# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot for Linear Regression ---
sns.scatterplot(x=y_test, y=y_pred_lin, alpha=0.6, ax=axes[0])
# Add a diagonal line representing a perfect prediction
axes[0].plot([0, 5], [0, 5], color='red', linestyle='--')
axes[0].set_xlabel("Actual Values (Median House Price)", fontsize=12)
axes[0].set_ylabel("Predicted Values", fontsize=12)
axes[0].set_title(f"Linear Regression\nRMSE: {rmse_lin:.4f} | R²: {r2_lin:.4f}", fontsize=14)
axes[0].grid(True)

# --- Plot for Ridge Regression ---
sns.scatterplot(x=y_test, y=y_pred_ridge, alpha=0.6, ax=axes[1])
# Add a diagonal line representing a perfect prediction
axes[1].plot([0, 5], [0, 5], color='red', linestyle='--')
axes[1].set_xlabel("Actual Values (Median House Price)", fontsize=12)
axes[1].set_ylabel("Predicted Values", fontsize=12)
axes[1].set_title(f"Ridge Regression\nRMSE: {rmse_ridge:.4f} | R²: {r2_ridge:.4f}", fontsize=14)
axes[1].grid(True)

# Add a main title for the entire figure
fig.suptitle("Model Performance: Actual vs. Predicted House Prices", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the suptitle
plt.show()