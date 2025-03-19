# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# Load the dataset
data = pd.read_csv('/content/gdrive/MyDrive/Mypbl.csv')
# Display the first few rows of the dataset
print(data.head())
# Encode categorical variables (Soil_Type, Crop_Type) for model input
le_soil = LabelEncoder()
le_crop = LabelEncoder()
data['Soil_Type'] = le_soil.fit_transform(data['Soil_Type'])
data['Crop_Type'] = le_crop.fit_transform(data['Crop_Type'])
# Split data into features (X) and target (y)
X = data.drop('Yield', axis=1)
y = data['Yield']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
# We will average the predictions from both models
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)
# Average predictions from both models
y_pred_combined = (y_pred_rf + y_pred_lr) / 2
# Recommendations for soil health, water, and sustainability
sustainability_recommendations = {
 'Wheat': {
 'soil_health': 'Wheat helps prevent soil erosion due to its dense root system. Consider
rotating with legumes to fix nitrogen.',
 'water_conservation': 'Wheat is a moderate water consumer. Use drip irrigation to optimize
water usage.',
 'sustainability': 'Minimize fertilizer usage by incorporating organic matter into the soil.'
 },
 'Rice': {
 'soil_health': 'Rice can lead to soil compaction. Alternate with dry crops like pulses to
restore soil structure.',
 'water_conservation': 'Rice is water-intensive. Consider using Alternate Wetting and Drying
(AWD) to save water.',
# MODEL 1: Random Forest Regressor
# MODEL 2: Linear Regression
# COMBINING BOTH MODELS
 'sustainability': 'Promote organic farming practices to reduce the environmental impact of
paddy fields.'
 },
 'Corn': {
 'soil_health': 'Corn depletes soil nutrients quickly. Use cover crops or rotate with nitrogenfixing plants.',
 'water_conservation': 'Corn requires moderate water. Use efficient irrigation systems to
reduce water use.',
 'sustainability': 'Practice minimal tillage and incorporate crop residues to promote soil
health.'
 }
}
# Function to provide recommendations based on crop
def sustainability_analysis(crop_name):
 if crop_name in sustainability_recommendations:
 recs = sustainability_recommendations[crop_name]
 print(f"\n--- Recommendations for {crop_name} ---")
 print(f"1. Soil Health: {recs['soil_health']}")
 print(f"2. Water Conservation: {recs['water_conservation']}")
 print(f"3. Sustainability: {recs['sustainability']}\n")
 else:
 print(f"No sustainability recommendations available for {crop_name}.")
# Function to predict best crop or check crop suitability based on user input
def predict_or_verify_crop():
 print("\nDo you want to input 'soil details' or 'crop'?\n")
 choice = input("Enter 'soil' for soil details or 'crop' for specific crop: ").lower()
 if choice == 'soil':
 # If user wants to input soil details
 soil_type = input("Enter Soil Type (Clay/Sandy/Loam): ").capitalize()
 ph_level = float(input("Enter pH Level of the Soil (e.g., 6.5): "))
 rainfall = float(input("Enter Rainfall in mm (e.g., 1000): "))
 temperature = float(input("Enter Temperature in Celsius (e.g., 25): "))
 fertilizer_usage = float(input("Enter Fertilizer Usage in kg/ha (e.g., 50): "))
 # Convert soil type to encoded value
 soil_type_encoded = le_soil.transform([soil_type])[0]
 # Prepare input data for prediction
 input_data = pd.DataFrame({
 'Soil_Type': [soil_type_encoded],
 'pH_Level': [ph_level],
 'Rainfall': [rainfall],
 'Temperature': [temperature],
 'Fertilizer_Usage': [fertilizer_usage]
 })
 # Predict yield for each crop type using both models
 best_crop = None
 best_yield = 0
 for crop in data['Crop_Type'].unique():
 input_data['Crop_Type'] = crop
 # Predict using both Random Forest and Linear Regression
 pred_rf = rf_model.predict(input_data)[0]
 pred_lr = lr_model.predict(input_data)[0]
 # Combined prediction (average)
 predicted_yield = (pred_rf + pred_lr) / 2
 if predicted_yield > best_yield:
 best_yield = predicted_yield
 best_crop = crop
 best_crop_name = le_crop.inverse_transform([best_crop])[0]
 print(f"\nBased on the soil details, the best crop to cultivate is: {best_crop_name}")
 print(f"Expected Yield: {best_yield:.2f} tons/hectare\n")
# Provide sustainability analysis
 sustainability_analysis(best_crop_name)
 elif choice == 'crop':
 # If user wants to verify a specific crop
 crop_type = input("Enter Crop Type (Wheat/Rice/Corn): ").capitalize()
 soil_type = input("Enter Soil Type (Clay/Sandy/Loam): ").capitalize()
 ph_level = float(input("Enter pH Level of the Soil (e.g., 6.5): "))
 rainfall = float(input("Enter Rainfall in mm (e.g., 1000): "))
 temperature = float(input("Enter Temperature in Celsius (e.g., 25): "))
 fertilizer_usage = float(input("Enter Fertilizer Usage in kg/ha (e.g., 50): "))
 # Convert soil type and crop type to encoded values
 soil_type_encoded = le_soil.transform([soil_type])[0]
 crop_type_encoded = le_crop.transform([crop_type])[0]
 # Prepare input data for prediction
 input_data = pd.DataFrame({
 'Soil_Type': [soil_type_encoded],
 'pH_Level': [ph_level],
 'Rainfall': [rainfall],
 'Temperature': [temperature],
 'Fertilizer_Usage': [fertilizer_usage],
 'Crop_Type': [crop_type_encoded]
 })
 # Predict yield for the given crop and soil conditions using both models
 pred_rf = rf_model.predict(input_data)[0]
 pred_lr = lr_model.predict(input_data)[0]
 # Combined prediction (average)
 predicted_yield = (pred_rf + pred_lr) / 2
 print(f"\nFor the crop {crop_type}, expected yield is: {predicted_yield:.2f} tons/hectare\n")
 # Provide sustainability analysis
 sustainability_analysis(crop_type)
# Call the function to start prediction or verification
predict_or_verify_crop()
# Optional: Evaluate model performance on the test set
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_combined = mean_squared_error(y_test, y_pred_combined)
print(f"Mean Squared Error of Random Forest: {mse_rf}")
print(f"Mean Squared Error of Linear Regression: {mse_lr}")
print(f"Mean Squared Error of Combined Model: {mse_combined}")
# Plot predicted vs actual values (for visualization)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_combined)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Crop Yield (Combined Model)')
plt.show()
