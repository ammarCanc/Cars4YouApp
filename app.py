import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from category_encoders import TargetEncoder

app = Flask(__name__)

# ------------------------------------------------------------------------
# 1. Load Artifacts
# ------------------------------------------------------------------------
try:
    # Use the filename you created in your notebook
    artifacts = joblib.load("car_assets.joblib")
    
    # Unpack all artifacts
    model = artifacts["model"]
    substring_maps = artifacts["substring_maps"]
    rare_maps = artifacts["rare_maps"]
    num_imputer = artifacts["num_imputer"]
    hierarchical_maps = artifacts["hierarchical_maps"]
    cat_imputer = artifacts["cat_imputer"]
    cat_cols = artifacts["cat_cols"]
    te = artifacts["te"]
    scaler = artifacts["scaler"]
    num_cols_new = artifacts["num_cols_new"]
    top_features = artifacts["top_features"]
    
    print("✅ Artifacts loaded successfully!")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    model = None

# ------------------------------------------------------------------------
# 2. Helper Functions (Copied from Notebook Logic)
# ------------------------------------------------------------------------

valid_zero_cols = ['previousOwners', 'hasDamage', 'tax']

def fix_negative_values(df, num_cols, target_col='price', threshold=0.7):
    df = df.copy()
    for col in num_cols:
        if col.lower() == target_col.lower(): continue
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df

def replace_zero_with_nan(df, num_cols):
    df = df.copy()
    for col in num_cols:
        if col not in df.columns or col in valid_zero_cols: continue
        if col == 'engineSize' and 'fuelType' in df.columns:
            mask = (df[col] == 0) & (df['fuelType'].str.lower() != 'electric')
        else:
            mask = (df[col] == 0)
        if mask.sum() > 0: df.loc[mask, col] = np.nan
    return df

def fix_year_values(df):
    if 'year' in df.columns:
        df['year'] = np.floor(df['year'])
        # Safety for future dates
        df.loc[df['year'] > 2026, 'year'] = np.nan
    return df

def fix_engine_size_values(df):
    if 'engineSize' in df.columns and 'fuelType' in df.columns:
        df.loc[(df['engineSize'] < 0.6) & (df['fuelType'].str.lower() != 'electric'), 'engineSize'] = np.nan
    return df

def apply_numerical_data_cleaning(df, num_cols):
    df = fix_negative_values(df, num_cols)
    df = replace_zero_with_nan(df, num_cols)
    df = fix_year_values(df)
    df = fix_engine_size_values(df)
    return df

def normalize_text(df, columns=None):
    df = df.copy()
    columns = columns or df.select_dtypes(include="object").columns
    for col in columns:
        df[col] = (df[col].astype(str).str.lower().str.strip()
                   .str.replace(r"\s+", " ", regex=True)
                   .str.replace(".", "", regex=False)
                   .replace(["nan", "none", "na", "n/a", "", "null", "unknown"], "unknown"))
    return df

def apply_substring_maps(df, maps):
    df = df.copy()
    for col, mapping in maps.items():
        if col in df.columns and mapping: df[col] = df[col].replace(mapping)
    return df

def apply_rare_category_maps(df, maps, other_label="other"):
    df = df.copy()
    for col, rare in maps.items():
        if col in df.columns and rare: df[col] = df[col].replace(rare, other_label)
    return df

def preprocess_text_test(df, substring_maps, rare_maps):
    df = normalize_text(df)
    df = apply_substring_maps(df, substring_maps)
    df = apply_rare_category_maps(df, rare_maps)
    return df

def clean_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()
    return df

def apply_numeric_imputer(df, num_cols, imputer):
    df = df.copy()
    # Only impute columns that exist in both df and num_cols
    cols_to_impute = [c for c in num_cols if c in df.columns]
    if cols_to_impute: 
        df[cols_to_impute] = imputer.transform(df[cols_to_impute])
    return df

def apply_hierarchical_group_modes(df, hierarchical_maps):
    df = df.copy()
    for col, levels in hierarchical_maps.items():
        missing = df[col].isnull() | (df[col] == "unknown")
        if not missing.any(): continue
        for deps, mapping in levels:
            if deps:
                if isinstance(deps, str): deps = [deps]
                if not all(d in df.columns for d in deps): continue
                try:
                    filled = df.loc[missing, deps].merge(
                        mapping.rename(col), left_on=deps, right_index=True, how="left")[col]
                    df.loc[missing, col] = df.loc[missing, col].fillna(filled)
                except Exception: pass
            else:
                df.loc[missing, col] = df.loc[missing, col].fillna(mapping)
            missing = df[col].isnull() | (df[col] == "unknown")
            if not missing.any(): break
    return df

def apply_categorical_imputer(df, cat_cols, imputer):
    df = df.copy()
    cols_to_impute = [c for c in cat_cols if c in df.columns]
    if cols_to_impute: df[cols_to_impute] = imputer.transform(df[cols_to_impute])
    return df

def preprocess_test(X_test, num_cols, num_imputer, hierarchical_maps, cat_imputer, cat_cols):
    X_test = clean_categoricals(X_test)
    X_test = apply_numeric_imputer(X_test, num_cols, num_imputer)
    X_test = apply_hierarchical_group_modes(X_test, hierarchical_maps)
    X_test = apply_categorical_imputer(X_test, cat_cols, cat_imputer)
    return X_test

def add_features(df):
    df = df.copy()
    # Ensure year consistency with Notebook
    current_year = datetime.now().year + 1 
    if 'year' in df.columns: 
        df['age'] = current_year - df['year']
    
    if 'mileage' in df.columns and 'age' in df.columns:
        # Avoid division by zero
        df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        
    if 'engineSize' in df.columns and 'mpg' in df.columns:
        df['efficiency_ratio'] = np.log1p(df['mpg'] / (df['engineSize'] + 1))
        
    return df

def drop_highly_correlated_features(df):
    cols_to_drop = ['year', 'mileage'] 
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

def align_features(df, feature_list):
    # This mimics the notebook's align_features but ensures we don't lose data
    # It adds missing columns with 0, but doesn't delete extra ones yet.
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    return df

# ------------------------------------------------------------------------
# 3. Main Prediction Function (Your exact notebook implementation logic)
# ------------------------------------------------------------------------
def predict_single_entry(input_dict, model, substring_maps, rare_maps, num_imputer, 
                         hierarchical_maps, cat_imputer, cat_cols, te, scaler, 
                         num_cols_new, top_features):
    test_df = pd.DataFrame([input_dict])

    # Fix: Ensure Brand is capitalized (often a mismatch source)
    if 'brand' in test_df.columns:
        test_df.rename(columns={'brand': 'Brand'}, inplace=True)
    
    # 1. Define num_cols_raw (Explicit list to match Imputer training)
    raw_num_cols = [
        'year', 'mileage', 'tax', 'mpg', 'engineSize', 
        'paintQuality%', 'previousOwners', 'hasDamage'
    ]
    
    # 2. Align Features (Ensure these columns exist)
    test_df = align_features(test_df, raw_num_cols)
    
    # 3. Apply Numerical Cleaning
    test_df = apply_numerical_data_cleaning(test_df, raw_num_cols)
    
    # 4. Text Preprocessing
    test_df = preprocess_text_test(
        test_df,
        substring_maps=substring_maps,
        rare_maps=rare_maps
    )
    
    # 5. General Preprocessing (Impute)
    test_df = preprocess_test(
        X_test=test_df,
        num_cols=raw_num_cols, 
        num_imputer=num_imputer,
        hierarchical_maps=hierarchical_maps,
        cat_imputer=cat_imputer,
        cat_cols=cat_cols
    )
    
    # 6. Target Encoding
    cat_cols_te = test_df.select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols_te:
        try:
            X_test_te = te.transform(test_df[cat_cols_te])
            X_test_te.index = test_df.index
            test_df = test_df.drop(columns=cat_cols_te)
            test_df = pd.concat([test_df, X_test_te], axis=1)
        except Exception as e:
            print(f"Target Encoding Warning: {e}")

    # 7. Feature Engineering
    test_df = add_features(test_df)
    test_df = drop_highly_correlated_features(test_df)
    
    # 8. Scale
    # Safety: Ensure scaler columns exist
    test_df = align_features(test_df, num_cols_new)
    test_df[num_cols_new] = scaler.transform(test_df[num_cols_new])
    
    # 9. Predict
    y_pred_log = model.predict(test_df[top_features])
    y_pred = np.expm1(y_pred_log)
    
    return y_pred[0]

# ------------------------------------------------------------------------
# 4. Flask Routes
# ------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    input_data = {}

    try:
        # Collect Inputs with Defaults
        input_data = {
            "brand": request.form.get('brand', 'Unknown'),
            "model": request.form.get('model', 'Unknown'),
            "year": float(request.form.get('year', 2018)),
            "mileage": float(request.form.get('mileage', 50000)),
            "engineSize": float(request.form.get('engineSize', 1.6)),
            "fuelType": request.form.get('fuelType', 'Petrol'),
            "transmission": request.form.get('transmission', 'Manual'),
            "tax": float(request.form.get('tax', 150)),
            "mpg": float(request.form.get('mpg', 50)),
            "hasDamage": float(request.form.get('hasDamage', 0)),
            "paintQuality%": float(request.form.get('paint', 90)),
            "previousOwners": float(request.form.get('owners', 1))
        }

        if model is None:
            return render_template('index.html', result="Error: Model not loaded.", original_input=input_data)

        price = predict_single_entry(
            input_dict=input_data,
            model=model,
            substring_maps=substring_maps,
            rare_maps=rare_maps,
            num_imputer=num_imputer,
            hierarchical_maps=hierarchical_maps,
            cat_imputer=cat_imputer,
            cat_cols=cat_cols,
            te=te,
            scaler=scaler,
            num_cols_new=num_cols_new,
            top_features=top_features
        )

        return render_template('index.html', result=f"£{price:,.2f}", original_input=input_data)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', result=f"Error: {str(e)}", original_input=input_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    # Check if we are local (if PORT is 5000, we are likely local)
    # This turns on the debugger ONLY when running locally
    debug_mode = (port == 5000) 
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)