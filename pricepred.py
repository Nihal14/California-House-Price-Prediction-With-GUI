import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score




# Step 1: Data Collection
dataset = pd.read_csv('housing.csv')

# Step 2: Data Preprocessing
X = dataset.drop('median_house_value', axis=1)
y = dataset['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['ocean_proximity'])
    ],
    remainder='passthrough'
)

X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_transformed)
X_test_scaled = scaler.transform(X_test_transformed)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Step 4: Model Selection
linear_regression = LinearRegression()
random_forest = RandomForestRegressor()

# Step 5: Model Training and Evaluation
linear_regression.fit(X_train_imputed, y_train)
random_forest.fit(X_train_imputed, y_train)

# Step 7: Model Validation
y_pred_lr = linear_regression.predict(X_test_imputed)
y_pred_rf = random_forest.predict(X_test_imputed)

# Calculate RMSE
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Calculate R-squared score
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# Determine the better model based on RMSE
better_model = 'Linear Regression' if rmse_lr < rmse_rf else 'Random Forest'

# Create a GUI window
window = tk.Tk()

# Define functions for button actions
def show_predicted_values():
    index = 1  # Specify the index of the data point for which you want to see the predicted value
    predicted_lr = linear_regression.predict(X_test_imputed[index].reshape(1, -1))
    predicted_rf = random_forest.predict(X_test_imputed[index].reshape(1, -1))

    predicted_values_str = "Linear Regression Predicted Value: " + str(predicted_lr[0]) + "\n"
    predicted_values_str += "Random Forest Predicted Value: " + str(predicted_rf[0])

    messagebox.showinfo("Predicted Values", predicted_values_str)



def show_linear_regression_plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.show()

def show_random_forest_plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest: Actual vs Predicted')
    plt.show()

def show_model_performance():
    performance_str = f"Linear Regression - RMSE: {rmse_lr}, R2 Score: {r2_lr}\n"
    performance_str += f"Random Forest - RMSE: {rmse_rf}, R2 Score: {r2_rf}\n"
    performance_str += f"Better Model: {better_model}"
    tk.messagebox.showinfo("Model Performance", performance_str)

# Create buttons
predicted_values_button = tk.Button(window, text='Show Predicted Values', command=show_predicted_values)
lr_plot_button = tk.Button(window, text='Show Linear Regression Plot', command=show_linear_regression_plot)
rf_plot_button = tk.Button(window, text='Show Random Forest Plot', command=show_random_forest_plot)
performance_button = tk.Button(window, text='Show Model Performance', command=show_model_performance)

# Place buttons in the window
predicted_values_button.pack()
lr_plot_button.pack()
rf_plot_button.pack()
performance_button.pack()

# Start the GUI event loop
window.mainloop()
