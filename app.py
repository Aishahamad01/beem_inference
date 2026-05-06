# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# from scipy.sparse import hstack
# import os

# app = Flask(__name__)


# MODELS_PATH = "models"


# print("Loading models and transformers...")
# model = joblib.load(os.path.join(MODELS_PATH, 'xgboost_model.pkl'))
# tfidf = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'))
# scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
# le = joblib.load(os.path.join(MODELS_PATH, 'label_encoder.pkl'))

# BEHAVIOR_COLS = [
#     'hour', 'is_weekend', 'src_msgs_per_min', 'src_msgs_per_hr', 'npdus', 'status',
#     'unique_dest_hr', 'spread_factor', 'charge', 'outconnector_encoded',
#     'unique_content_per_src_hr', 'unique_src_hr', 'dest_time_diff_avg'
# ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data provided"}), 400

#         df = pd.DataFrame([data])

        
#         try:
#             df['outconnector_encoded'] = le.transform(df['outconnector'].astype(str))
#         except ValueError:
#             # Handle unseen categories by using a default (usually 0 or mode)
#             df['outconnector_encoded'] = 0 

#         # --- 2. Behavioral Features ---
#         X_beh = df[BEHAVIOR_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
#         X_beh_scaled = scaler.transform(X_beh)

#         text_content = [str(data.get('sms_content', ""))]
#         X_tfidf = tfidf.transform(text_content)

        
#         X_final = hstack([X_beh_scaled, X_tfidf]).tocsr()

        
#         prediction = model.predict(X_final)[0]
#         probability = model.predict_proba(X_final)[0][1]

#         result = {
#             "prediction": int(prediction),
#             "label": "AIT Fraud" if prediction == 1 else "Normal",
#             "probability": round(float(probability), 4)
#         }

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # Run the app on localhost:5000
#     app.run(debug=True, port=5000)


import os
import sys
import platform
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from scipy.sparse import hstack


# if platform.system() == "Darwin": 
#     os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

app = Flask(__name__)

MODELS_PATH = "models"

def load_models():
    try:
        model = joblib.load(os.path.join(MODELS_PATH, 'xgboost_model.pkl'))
        tfidf = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'))
        scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
        le = joblib.load(os.path.join(MODELS_PATH, 'label_encoder.pkl'))
        print("All models and transformers loaded successfully.")
        return model, tfidf, scaler, le
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

model, tfidf, scaler, le = load_models()

# ==========================================
# FEATURE ENGINEERING LOGIC
# ==========================================
def engineer_features(df):
    """
    Converts raw SMS logs into the behavioral features the model expects.
    """
    # 1. Time based features
    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
    df['hour'] = df['starttime'].dt.hour.fillna(0).astype(int)
    df['is_weekend'] = df['starttime'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Aggregation Helper Columns
    df['date_min'] = df['starttime'].dt.to_period('min')
    df['date_hour'] = df['starttime'].dt.to_period('h')
    
    # 3. Behavioral Frequencies (Calculated based on the current batch)
    df['src_msgs_per_min'] = df.groupby(['date_min', 'sourceaddr'])['msgid'].transform('count')
    df['src_msgs_per_hr'] = df.groupby(['date_hour', 'sourceaddr'])['msgid'].transform('count')
    df['unique_dest_hr'] = df.groupby(['date_hour', 'sourceaddr'])['destaddr'].transform('nunique')
    df['unique_src_hr'] = df.groupby('date_hour')['sourceaddr'].transform('nunique')
    df['unique_content_per_src_hr'] = df.groupby(['date_hour', 'sourceaddr'])['sms_content'].transform('nunique')
    
    # 4. Derived Factor
    df['spread_factor'] = df['unique_dest_hr'] / (df['src_msgs_per_hr'] + 1)
    
    # 5. Time Difference (Standardized to 0 for single-message requests)
    df_sorted = df.sort_values(['destaddr', 'starttime'])
    df_sorted['dest_time_diff'] = df_sorted.groupby('destaddr')['starttime'].diff().dt.total_seconds() / 3600.0
    df['dest_time_diff_avg'] = df_sorted.groupby('destaddr')['dest_time_diff'].transform('mean').fillna(0)
    
    # 6. Label Encoding
    try:
        df['outconnector_encoded'] = le.transform(df['outconnector'].astype(str))
    except:
        # Handle unseen outconnectors
        df['outconnector_encoded'] = 0

    # Ensure numeric types
    target_cols = [
        'hour', 'is_weekend', 'src_msgs_per_min', 'src_msgs_per_hr','npdus','status',
        'unique_dest_hr', 'spread_factor', 'charge', 'outconnector_encoded',
        'unique_content_per_src_hr', 'unique_src_hr', 'dest_time_diff_avg'
    ]
    
    X_beh = df[target_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X_beh, df['sms_content'].astype(str).tolist()

# ==========================================
# API ENDPOINT
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No data"}), 400
        
        # Accept either a single dict or a list of dicts
        if isinstance(input_data, dict):
            input_data = [input_data]
            
        df_raw = pd.DataFrame(input_data)
        
        # Verify columns exist (fill missing with empty string/0)
        required_cols = ['msgid','starttime','npdus','status','charge','sourceaddr','destaddr','outconnector','sms_content']
        for col in required_cols:
            if col not in df_raw.columns:
                df_raw[col] = "" if col in ['starttime', 'sourceaddr', 'destaddr', 'outconnector', 'sms_content'] else 0

        # 1. Feature Engineering
        X_beh, text_list = engineer_features(df_raw)
        
        # 2. Scaling
        X_beh_scaled = scaler.transform(X_beh)
        
        # 3. TF-IDF
        X_tfidf = tfidf.transform(text_list)
        
        # 4. Combine
        X_final = hstack([X_beh_scaled, X_tfidf]).tocsr()
        
        # 5. Predict
        preds = model.predict(X_final)
        probs = model.predict_proba(X_final)[:, 1]
        
        # 6. Prepare Response
        results = []
        for i in range(len(preds)):
            results.append({
                "msgid": str(df_raw.iloc[i]['msgid']),
                "prediction": int(preds[i]),
                "label": "AIT Fraud" if preds[i] == 1 else "Normal",
                "probability": round(float(probs[i]), 4)
            })
            
        return jsonify(results)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)