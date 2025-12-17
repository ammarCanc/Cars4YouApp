from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from category_encoders import TargetEncoder

app = Flask(__name__)
assets = joblib.load('car_assets.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        # 1. Capture Form Data
        data = {
            'Brand': [request.form.get('brand')],
            'model': [request.form.get('model')],
            'year': [float(request.form.get('year'))],
            'mileage': [float(request.form.get('mileage'))],
            'fuelType': [request.form.get('fuelType')],
            'tax': [float(request.form.get('tax'))],
            'mpg': [float(request.form.get('mpg'))],
            'engineSize': [float(request.form.get('engineSize'))],
            'paintQuality%': [float(request.form.get('paint'))],
            'previousOwners': [float(request.form.get('owners'))],
            'hasDamage': [float(request.form.get('hasDamage', 0))]
        }
        df = pd.DataFrame(data)

        # 2. Map/Replace Logic
        current_model = str(df['model'].iloc[0])
        for key, val in assets['substring_maps'].items():
            if key in current_model:
                df.at[0, 'model'] = val
                break
        df['model'] = df['model'].replace(assets['rare_maps'])

        # 3. Impute & Encode
        # Fill missing values for the 8 numerical columns
        df[assets['num_cols']] = assets['num_imputer'].transform(df[assets['num_cols']])
        # Transform categories to numbers
        df_encoded = assets['te'].transform(df)

        # 4. THE FIX: Filter down to exactly the 4 features the scaler knows
        # This resolves: "Unexpected input dimension 11, expected 4"
        df_selected = df_encoded[assets['top_features']]
        
        # 5. Scale & Predict
        df_final = assets['scaler'].transform(df_selected)
        prediction = assets['model'].predict(df_final)[0]

        return render_template('index.html', result=f"£{prediction:,.2f}")

    except Exception as e:
        # This will show you exactly which step is failing in the UI
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)