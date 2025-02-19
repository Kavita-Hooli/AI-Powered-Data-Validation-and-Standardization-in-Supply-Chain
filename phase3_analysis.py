import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Use regressor for continuous target
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.preprocessing import StandardScalerpython
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("supply_chain_data.csv")

# Separate numerical and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Apply KNN Imputer only on numerical data
imputer = KNNImputer(n_neighbors=5)
data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)

# Combine back categorical columns
data_cleaned = pd.concat([data_numeric_imputed, data[categorical_cols]], axis=1)

# Apply One-Hot Encoding on categorical columns
data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_cols)

# Check for missing values
print("Missing Values After Imputation:")
print(data_cleaned.isnull().sum())

# Apply Isolation Forest for anomaly detection
iso = IsolationForest(contamination=0.01, random_state=42)

# Fit on numerical columns only
data_cleaned["Anomaly"] = iso.fit_predict(data_cleaned[numeric_cols])

# Remove detected outliers
data_cleaned = data_cleaned[data_cleaned["Anomaly"] == 1].drop(columns=["Anomaly"])

# Print the number of outliers removed
print(f"Number of Outliers Removed: {len(data) - len(data_cleaned)}")

# Set the feature columns (X) and target column (y)
X = data_cleaned.drop(columns=["Revenue generated"])  # Replace 'Revenue generated' with your actual target column name
y = data_cleaned["Revenue generated"]  # Set the target column

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor model
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Set the feature columns (X) and target column (y)
X = data_cleaned.drop(columns=["Revenue generated"])  # Replace with the actual target column name
y = data_cleaned["Revenue generated"]  # Set the target column

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor model
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R² for evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Apply log transformation to the target variable
y = np.log1p(y)  # Log(x + 1) to avoid issues with zeros or negative values

# Print the evaluation results
print(f"Decision Tree Regressor MSE: {mse}")
print(f"Decision Tree Regressor R²: {r2}")

# Train a Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions and calculate MSE and R²
y_rf_pred = rf_regressor.predict(X_test)
mse_rf = mean_squared_error(y_test, y_rf_pred)
r2_rf = r2_score(y_test, y_rf_pred)

# Print results
print(f"Random Forest Regressor MSE: {mse_rf}")
print(f"Random Forest Regressor R²: {r2_rf}")

# Train a Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(random_state=42)
gb_regressor.fit(X_train, y_train)

# Make predictions and calculate MSE and R²
y_gb_pred = gb_regressor.predict(X_test)
mse_gb = mean_squared_error(y_test, y_gb_pred)
r2_gb = r2_score(y_test, y_gb_pred)

# Print results
print(f"Gradient Boosting Regressor MSE: {mse_gb}")
print(f"Gradient Boosting Regressor R²: {r2_gb}")

# Apply StandardScaler to normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate Decision Tree Regressor
print("Decision Tree Regressor Performance:")
y_dt_pred = regressor.predict(X_test)
mse_dt = mean_squared_error(y_test, y_dt_pred)
r2_dt = r2_score(y_test, y_dt_pred)
print(f"MSE: {mse_dt}")
print(f"R²: {r2_dt}")

# Evaluate Random Forest Regressor
print("\nRandom Forest Regressor Performance:")
y_rf_pred = rf_regressor.predict(X_test)
mse_rf = mean_squared_error(y_test, y_rf_pred)
r2_rf = r2_score(y_test, y_rf_pred)
print(f"MSE: {mse_rf}")
print(f"R²: {r2_rf}")

# Evaluate Gradient Boosting Regressor
print("\nGradient Boosting Regressor Performance:")
y_gb_pred = gb_regressor.predict(X_test)
mse_gb = mean_squared_error(y_test, y_gb_pred)
r2_gb = r2_score(y_test, y_gb_pred)
print(f"MSE: {mse_gb}")
print(f"R²: {r2_gb}")

# Initialize the Linear Regression model
lr_regressor = LinearRegression()

# Train the Linear Regression model
lr_regressor.fit(X_train, y_train)

# Make predictions
y_lr_pred = lr_regressor.predict(X_test)

# Calculate MSE and R²
mse_lr = mean_squared_error(y_test, y_lr_pred)
r2_lr = r2_score(y_test, y_lr_pred)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R²: {r2_lr}")

# Evaluate Linear Regression (if you tried it)
print("\nLinear Regression Performance:")
y_lr_pred = lr_regressor.predict(X_test)
mse_lr = mean_squared_error(y_test, y_lr_pred)
r2_lr = r2_score(y_test, y_lr_pred)
print(f"MSE: {mse_lr}")
print(f"R²: {r2_lr}")

# Initialize XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(random_state=42)

# Train the XGBoost Regressor model
xgb_regressor.fit(X_train, y_train)

# Make predictions
y_xgb_pred = xgb_regressor.predict(X_test)

# Calculate MSE and R²
mse_xgb = mean_squared_error(y_test, y_xgb_pred)
r2_xgb = r2_score(y_test, y_xgb_pred)

# Print the results for XGBoost
print(f"XGBoost Regressor MSE: {mse_xgb}")
print(f"XGBoost Regressor R²: {r2_xgb}")

# Evaluate XGBoost Regressor (if you tried it)
print("\nXGBoost Regressor Performance:")
y_xgb_pred = xgb_regressor.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_xgb_pred)
r2_xgb = r2_score(y_test, y_xgb_pred)
print(f"MSE: {mse_xgb}")
print(f"R²: {r2_xgb}")

# Store model names and their MSE
models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Linear Regression', 'XGBoost']
mse_values = [mse_dt, mse_rf, mse_gb, mse_lr, mse_xgb]

# Create a bar chart to compare MSE
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title("Model Comparison (MSE)")
plt.xlabel("Models")
plt.ylabel("MSE")
plt.show()