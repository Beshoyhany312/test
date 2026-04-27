
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

def load_and_predict_lgbm(input_data):
    # Load the scaler
    scaler = joblib.load('scaler.joblib')

    # Load the LightGBM model
    lgbm_model = joblib.load('lightgbm_regressor_model.joblib')

    # Ensure input_data is a DataFrame with the correct column order
    input_df = pd.DataFrame([input_data], columns=['Site Area (square meters)', 'Water Consumption (liters/day)', 'Recycling Rate (%)', 'Utilisation Rate (%)', 'Air Quality Index (AQI)', 'Issue Resolution Time (hours)', 'Resident Count (number of people)', 'Structure Type_Industrial', 'Structure Type_Mixed-use', 'Structure Type_Residential'])
    
    # Preprocess the input data using the loaded scaler
    scaled_input_data = scaler.transform(input_df)

    # Make prediction
    prediction = lgbm_model.predict(scaled_input_data)
    return prediction[0]

if __name__ == '__main__':
    # Example usage (replace with actual input from Streamlit)
    # The order of features must match the training data
    # Assuming X.columns contains the feature names in the correct order
    # Example input, adjust based on your actual feature structure
    sample_input = [
        1360,  # 'Site Area (square meters)'
        2519.0, # 'Water Consumption (liters/day)'
        68,     # 'Recycling Rate (%)'
        59,     # 'Utilisation Rate (%)'
        51,     # 'Air Quality Index (AQI)'
        34,     # 'Issue Resolution Time (hours)'
        6,      # 'Resident Count (number of people)'
        False,  # 'Structure Type_Industrial'
        True,   # 'Structure Type_Mixed-use'
        False   # 'Structure Type_Residential'
    ]

    predicted_cost = load_and_predict_lgbm(sample_input)
    print(f"Predicted Electricity Cost (LightGBM): {predicted_cost:.2f} USD/month")

