import pandas as pd
import joblib
model = joblib.load('price_prediction_model.pkl')

def predict_crop_price(year, month_num, crop_name):
    crops = [
        'Coconut', 'Coffee', 'Cotton', 'Ginger(Dry)', 'Groundnut', 'Jowar(Sorghum)',
        'Maize', 'Millets', 'Rice', 'Sugar', 'Sugarcane', 'Sunflower',
        'Tea', 'Turmeric', 'Wheat'
    ]

    input_data = {
        'year': year,
        'month_num': month_num
    }

    for crop in crops:
        input_data[f'commodity_name_{crop}'] = 1 if crop == crop_name else 0

    input_df = pd.DataFrame([input_data])
    predicted_price = model.predict(input_df)[0]
    return round(predicted_price, 2)

price = predict_crop_price(year=2025, month_num=11, crop_name='Wheat')
print(f"Predicted avg_modal_price: ₹{price}")

