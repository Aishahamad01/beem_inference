import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from scipy.sparse import hstack
import sklearn

# Print version for debugging
print(f"DEBUG: Running Scikit-learn version {sklearn.__version__}")

app = Flask(__name__)

# Folder where your .pkl files are stored
MODELS_PATH = "models"

def load_models():
    try:
        # Load the artifacts
        model = joblib.load(os.path.join(MODELS_PATH, 'best_fraud_model.pkl'))
        tfidf = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'))
        scaler = joblib.load(os.path.join(MODELS_PATH, 'feature_scaler.pkl'))
        le = joblib.load(os.path.join(MODELS_PATH, 'label_encoder.pkl'))

        # --- VERSION MISMATCH PATCH ---
        # Fix for: 'LogisticRegression' object has no attribute 'multi_class'
        if not hasattr(model, 'multi_class'):
            model.multi_class = 'auto'
            print("⚠️ Applied patch: added missing 'multi_class' attribute to model.")
        
        # Ensure model knows it was trained on 14 features
        if not hasattr(model, 'n_features_in_'):
            # Note: This is the count of (Numeric Features + TF-IDF features)
            # If standard XGB/LogReg, it might need this to skip validation
            pass 

        print("✅ All models and transformers loaded and patched successfully.")
        return model, tfidf, scaler, le
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

# Initialize models
model, tfidf, scaler, le = load_models()

def engineer_features(df):
    """
    Aligned with training script: EXACTLY 14 Behavioral Features
    """
    df = df.copy()
    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
    
    # 1. Temporal features
    df['hour'] = df['starttime'].dt.hour.fillna(0).astype(int)
    df['is_weekend'] = df['starttime'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0).astype(int)
    
    # 2. Aggregation Periods
    df['date_hour'] = df['starttime'].dt.to_period('h')
    df['date_day']  = df['starttime'].dt.to_period('D')
    
    # 3. Behavioral Counts (Calculated per batch)
    df['src_msgs_per_hr'] = df.groupby(['date_hour', 'sourceaddr'])['msgid'].transform('count')
    df['src_msgs_per_day'] = df.groupby(['date_day', 'sourceaddr'])['msgid'].transform('count')
    df['dest_msgs_per_hr']  = df.groupby(['date_hour', 'destaddr'])['msgid'].transform('count')
    df['dest_msgs_per_day']  = df.groupby(['date_day', 'destaddr'])['msgid'].transform('count')
    
    # 4. Diversity/Uniqueness
    df['unique_dest_day'] = df.groupby(['date_day', 'sourceaddr'])['destaddr'].transform('nunique')
    df['unique_src_day'] = df.groupby('date_day')['sourceaddr'].transform('nunique')
    df['unique_content_per_src_day'] = df.groupby(['date_day', 'sourceaddr'])['sms_content'].transform('nunique')
    
    # 5. Destination timing (Average time between messages to the same destination)
    df_sorted = df.sort_values(['destaddr', 'starttime'])
    df_sorted['dest_time_diff'] = df_sorted.groupby('destaddr')['starttime'].diff().dt.total_seconds() / 3600.0
    df['dest_time_diff_avg'] = df_sorted.groupby('destaddr')['dest_time_diff'].transform('mean').fillna(0)

    # 6. Label Encoding for Outconnector
    try:
        df['outconnector_encoded'] = le.transform(df['outconnector'].astype(str))
    except:
        # Fallback for outconnectors the encoder has never seen
        df['outconnector_encoded'] = 0

    # THE CRITICAL LIST: Order and count (14) must match Training exactly
    behavior_cols = [
        'hour', 
        'is_weekend', 
        'src_msgs_per_day', 
        'dest_msgs_per_hr', 
        'src_msgs_per_hr', 
        'dest_msgs_per_day', 
        'npdus', 
        'status', 
        'unique_dest_day', 
        'charge', 
        'outconnector_encoded', 
        'unique_content_per_src_day', 
        'unique_src_day', 
        'dest_time_diff_avg'
    ]
    
    # Fill any missing required columns with 0
    for col in behavior_cols:
        if col not in df.columns:
            df[col] = 0

    X_beh = df[behavior_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X_beh, df['sms_content'].astype(str).tolist()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No data received"}), 400
        
        # Accept single dictionary or list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]
            
        df_raw = pd.DataFrame(input_data)
        
        # 1. Feature Engineering
        X_beh, text_list = engineer_features(df_raw)
        
        # 2. Scaling (Use .values to bypass "feature names" strict check)
        X_beh_scaled = scaler.transform(X_beh.values)
        
        # 3. TF-IDF Vectorization
        X_tfidf = tfidf.transform(text_list)
        
        # 4. Combine Behavioral and Text Features
        X_final = hstack([X_beh_scaled, X_tfidf]).tocsr()
        
        # 5. Model Prediction
        preds = model.predict(X_final)
        probs = model.predict_proba(X_final)[:, 1]
        
        # 6. Prepare Response JSON
        results = []
        for i in range(len(preds)):
            results.append({
                "msgid": str(df_raw.iloc[i].get('msgid', 'N/A')),
                "prediction": int(preds[i]),
                "label": "AIT Fraud" if preds[i] == 1 else "Normal",
                "probability": round(float(probs[i]), 4)
            })
            
        return jsonify(results)

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e), 
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 to make it accessible inside Docker/Network
    app.run(debug=True, host='0.0.0.0', port=5000)