import pandas as pd
import matplotlib.pyplot as plt
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

# Step 7: Model Deployment
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR OCEAN'],
})

new_data_transformed = column_transformer.transform(new_data)
new_data_scaled = scaler.transform(new_data_transformed)
new_data_imputed = imputer.transform(new_data_scaled)

predicted_prices_lr = linear_regression.predict(new_data_imputed)
predicted_prices_rf = random_forest.predict(new_data_imputed)

print("Linear Regression - Predicted house price:")
print(predicted_prices_lr)
print("Random Forest - Predicted house price:")
print(predicted_prices_rf)

# Step 8: Plotting Actual vs Predicted Values
y_pred_lr = linear_regression.predict(X_test_imputed)
y_pred_rf = random_forest.predict(X_test_imputed)

plt.scatter(y_test, y_pred_lr, color='b', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

plt.scatter(y_test, y_pred_rf, color='b', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted')
plt.show()
