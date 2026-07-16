import os
import pandas as pd
import numpy as np
import joblib
import warnings
from flask import Flask, request, jsonify
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ====================== CONFIG ======================
MODELS_PATH = "models"
os.makedirs(MODELS_PATH, exist_ok=True)

# ====================== LOAD ASSETS ======================
try:
    model = joblib.load(os.path.join(MODELS_PATH, "mlp_classifier_model.pkl"))
    tfidf = joblib.load(os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_PATH, "label_encoder.pkl"))
    behavior_cols = joblib.load(os.path.join(MODELS_PATH, "behavior_cols.pkl"))
    
    print("✅ All preprocessing assets loaded successfully!")
    print(f"   • Model: RandomForest")
    print(f"   • Behavior Columns: {len(behavior_cols)}")
    print(f"   • TF-IDF Features: {tfidf.get_feature_names_out().shape[0]}")

except Exception as e:
    print(f"❌ Loading Error: {e}")
    raise


# ====================== FEATURE ENGINEERING (Aligned & Fixed) ======================
def engineer_features(df):
    df = df.copy()
    
    # Ensure standard essential baseline text columns exist
    for expected_col in ['msgid', 'sourceaddr', 'destaddr', 'sms_content']:
        if expected_col not in df.columns:
            df[expected_col] = ''
            
    if 'charge' not in df.columns:
        df['charge'] = 0.0

    # Time Features
    if 'starttime' not in df.columns:
        df['starttime'] = pd.Timestamp.now()
    else:
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce').fillna(pd.Timestamp.now())

    df['hour'] = df['starttime'].dt.hour.fillna(0).astype(int)
    df['is_weekend'] = df['starttime'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0).astype(int)

    df['date_min'] = df['starttime'].dt.to_period('min')
    df['date_hour'] = df['starttime'].dt.to_period('h')
    df['date_day'] = df['starttime'].dt.to_period('D')

    # =====================================================================
    # SMART METRIC PROTECTION (Fixes Training-Serving Skew)
    # Uses payload values if provided, otherwise calculates from batch.
    # =====================================================================
    def set_behavioral_metric(col_name, groupby_cols, agg_type='count', target_col='msgid'):
        if col_name not in df.columns or df[col_name].isnull().all():
            if agg_type == 'count':
                df[col_name] = df.groupby(groupby_cols)[target_col].transform('count')
            elif agg_type == 'nunique':
                df[col_name] = df.groupby(groupby_cols)[target_col].transform('nunique')
        else:
            # Fallback for empty rows inside an otherwise provided column
            if agg_type == 'count':
                fallback = df.groupby(groupby_cols)[target_col].transform('count')
            elif agg_type == 'nunique':
                fallback = df.groupby(groupby_cols)[target_col].transform('nunique')
            df[col_name] = df[col_name].fillna(fallback)

    # Resolve each metric either from the request payload or by grouping
    set_behavioral_metric('src_msgs_per_min', ['date_min', 'sourceaddr'], 'count')
    set_behavioral_metric('src_msgs_per_hr', ['date_hour', 'sourceaddr'], 'count')
    set_behavioral_metric('src_msgs_per_day', ['date_day', 'sourceaddr'], 'count')
    
    set_behavioral_metric('unique_dest_hr', ['date_hour', 'sourceaddr'], 'nunique', 'destaddr')
    set_behavioral_metric('unique_src_hr', ['date_hour'], 'nunique', 'sourceaddr')
    set_behavioral_metric('unique_content_per_src_hr', ['date_hour', 'sourceaddr'], 'nunique', 'sms_content')
    set_behavioral_metric('dest_msgs_per_src_hr', ['date_hour', 'sourceaddr', 'destaddr'], 'count')

    # Re-calculate Spread Factor smoothly
    if 'spread_factor' not in df.columns or df['spread_factor'].isnull().all():
        df['spread_factor'] = df['unique_dest_hr'] / (df['src_msgs_per_hr'] + 1)
    else:
        df['spread_factor'] = df['spread_factor'].fillna(df['unique_dest_hr'] / (df['src_msgs_per_hr'] + 1))

    # Time Difference Calculation
    if 'dest_time_diff_avg' not in df.columns or df['dest_time_diff_avg'].isnull().all():
        df_sorted = df.sort_values(['destaddr', 'starttime'])
        df_sorted['dest_time_diff'] = df_sorted.groupby('destaddr')['starttime'].diff().dt.total_seconds() / 3600.0
        df['dest_time_diff_avg'] = df_sorted.groupby('destaddr')['dest_time_diff'].transform('mean').fillna(0)
    else:
        df['dest_time_diff_avg'] = df['dest_time_diff_avg'].fillna(0)

    # Outconnector Encoding (Safe handling for unseen categories)
    df['outconnector'] = df.get('outconnector', 'unknown').astype(str)
    known_classes = set(label_encoder.classes_)
    fallback_val = int(label_encoder.transform(['unknown'])[0]) if 'unknown' in known_classes else 0
        
    df['outconnector_encoded'] = df['outconnector'].apply(
        lambda x: int(label_encoder.transform([x])[0]) if x in known_classes else fallback_val
    ).astype(int)

    # Ensure all behavior_cols exist
    for col in behavior_cols:
        if col not in df.columns:
            df[col] = 0

    X_beh = df[behavior_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
    
    return X_beh, df['sms_content'].astype(str).tolist()


# ====================== PREDICTION ENDPOINT ======================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Support single JSON objects or a list of JSON records
        if isinstance(data, dict):
            df_raw = pd.DataFrame([data])
        else:
            df_raw = pd.DataFrame(data)

        # Run feature engineering 
        X_beh, text_list = engineer_features(df_raw)
        
        # Scale and Vectorize
        X_beh_scaled = scaler.transform(X_beh)
        X_tfidf = tfidf.transform(text_list)
        
        # Combine Behavior and Text features
        X_final = hstack([X_beh_scaled, X_tfidf]).tocsr()
        
        # Predict
        predictions = model.predict(X_final)
        probs = model.predict_proba(X_final)[:, 1] if hasattr(model, "predict_proba") else None

        results = []
        for i in range(len(predictions)):
            results.append({
                "msgid": str(df_raw.iloc[i].get('msgid', f"msg_{i}")),
                "prediction": int(predictions[i]),
                "status": "Fraud (AIT)" if predictions[i] == 1 else "Normal",
                "probability": round(float(probs[i]), 4) if probs is not None else None
            })

        return jsonify({
            "status": "success",
            "predictions": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "RandomForest",
        "behavior_cols": len(behavior_cols),
        "tfidf_features": int(tfidf.get_feature_names_out().shape[0])
    })


if __name__ == '__main__':
    print("🚀 SMS Fraud Detection API Started (Aligned with Preprocessing)")
    app.run(host='0.0.0.0', port=5000, debug=True)