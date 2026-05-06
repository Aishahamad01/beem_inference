# SMS AIT Fraud Detection API

> A Machine Learning-powered Flask API for detecting **Artificially Inflated Traffic (AIT) Fraud** in SMS gateway logs. Transforms raw SMS logs into behavioral patterns and classifies them in real-time using an XGBoost model.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the API](#running-the-api)
- [API Reference](#api-reference)
- [Preprocessing Logic](#preprocessing-logic)
- [Testing with Postman](#testing-with-postman)
- [Troubleshooting](#troubleshooting)

---

## Features

- **On-the-fly Feature Engineering** — Accepts 21 raw input columns and automatically computes 13 behavioral metrics (frequencies, spread factors, time-diffs).
- **Text Analysis** — Integrates TF-IDF vectorization for SMS message content analysis.
- **Batch Inference** — Supports both single-message and multi-message (array) JSON inputs.
- **Cross-Platform Support** — Custom dependency handling for macOS (Apple Silicon & Intel) and Windows.

---

## Project Structure

```
beem_project/
├── app.py                   # Flask API entry point
├── models/                  # Pre-trained model artifacts
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python **3.9** or higher
- **macOS only**: [Homebrew](https://brew.sh/) (required for OpenMP)

### 1. System-Level Dependencies

**macOS (Silicon & Intel)**

XGBoost requires `libomp` for parallel processing on Mac:

```bash
brew install libomp
```

> After installing, restart your terminal before proceeding.

**Windows**

Ensure [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) is installed (typically present by default).

### 2. Install Python Dependencies

```bash
pip install flask joblib pandas xgboost lightgbm scikit-learn scipy
```

---

## Running the API

Navigate to the project directory and start the Flask server:

```bash
cd /your/path/to/beem_project
python app.py
```

On successful startup, you will see:

```
✅ All models and transformers loaded successfully.
 * Running on http://127.0.0.1:5000
```

The prediction endpoint is now live at:

```
POST http://127.0.0.1:5000/predict
```

---

## API Reference

### `POST /predict`

Classifies one or more SMS log entries as fraudulent (AIT) or legitimate.

**Request Headers**

| Header         | Value            |
|----------------|------------------|
| `Content-Type` | `application/json` |

**Request Body**

Send a JSON array of raw SMS log objects. Sending multiple messages in a single request is recommended for more accurate behavioral analysis (e.g., `messages_per_hour` calculations).

| Field           | Type    | Description                              |
|-----------------|---------|------------------------------------------|
| `msgid`         | string  | Unique message identifier                |
| `starttime`     | string  | Timestamp — format `YYYY-MM-DD HH:MM:SS` |
| `npdus`         | integer | Number of PDUs                           |
| `charcode`      | integer | Character encoding code                  |
| `status`        | integer | Delivery status code                     |
| `bufferedstatus`| integer | Buffered delivery status                 |
| `msgtype`       | integer | Message type identifier                  |
| `charge`        | float   | Charge applied to message                |
| `sourceaddr`    | string  | Sender address / shortcode               |
| `destaddr`      | string  | Destination MSISDN                       |
| `reason`        | integer | Failure reason code                      |
| `reasontext`    | string  | Human-readable reason                    |
| `note`          | string  | Optional notes                           |
| `smscid`        | string  | SMSC identifier                          |
| `inconnector`   | string  | Inbound connector name                   |
| `outconnector`  | string  | Outbound connector name                  |
| `mcc`           | string  | Mobile Country Code                      |
| `mnc`           | string  | Mobile Network Code                      |
| `dlr_mcc`       | string  | DLR Mobile Country Code                  |
| `dlr_mnc`       | string  | DLR Mobile Network Code                  |
| `sms_content`   | string  | Full text of the SMS message             |

**Example Request**

```json
[
    {
        "msgid": "ID_001",
        "starttime": "2025-08-20 10:15:00",
        "npdus": 1,
        "charcode": 0,
        "status": 1,
        "bufferedstatus": 0,
        "msgtype": 1,
        "charge": 0.0,
        "sourceaddr": "GoogleOTP",
        "destaddr": "447700112233",
        "reason": 0,
        "reasontext": "delivered",
        "note": "",
        "smscid": "smsc_01",
        "inconnector": "api_in",
        "outconnector": "primary_gateway",
        "mcc": "234",
        "mnc": "10",
        "dlr_mcc": "234",
        "dlr_mnc": "10",
        "sms_content": "G-554321 is your Google verification code."
    },
    {
        "msgid": "ID_002",
        "starttime": "2025-08-20 10:15:05",
        "npdus": 1,
        "charcode": 0,
        "status": 1,
        "bufferedstatus": 0,
        "msgtype": 1,
        "charge": 0.0,
        "sourceaddr": "GoogleOTP",
        "destaddr": "447700112244",
        "reason": 0,
        "reasontext": "delivered",
        "note": "",
        "smscid": "smsc_01",
        "inconnector": "api_in",
        "outconnector": "primary_gateway",
        "mcc": "234",
        "mnc": "10",
        "dlr_mcc": "234",
        "dlr_mnc": "10",
        "sms_content": "G-998877 is your Google verification code."
    }
]
```

---

## Preprocessing Logic

Before the model runs inference, the API applies the following transformations:

| Step | Description |
|------|-------------|
| **Temporal Extraction** | Parses `starttime` into `hour` and `is_weekend` features |
| **Frequency Counts** | Groups by `sourceaddr` to compute `messages_per_minute` and `messages_per_hour` |
| **Diversity Metric** | Calculates `spread_factor` = unique destinations / total messages |
| **Content Vectorization** | Transforms `sms_content` into a 500-dimension array using the saved TF-IDF model |
| **Categorical Encoding** | Converts `outconnector` to a numeric ID using the saved label encoder |

---

## Testing with Postman

1. Open Postman and create a new request.
2. Set the method to **POST**.
3. Enter the URL: `http://127.0.0.1:5000/predict`
4. Go to the **Body** tab → select **raw** → set the format to **JSON**.
5. Paste the example payload from the [API Reference](#api-reference) section above.
6. Click **Send**.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `libxgboost.dylib not loaded` | Missing OpenMP on macOS | Run `brew install libomp` and restart terminal |
| `ModuleNotFoundError` | Missing Python dependencies | Run `pip install flask joblib pandas xgboost lightgbm scikit-learn scipy` |
| `UserWarning: X does not have valid feature names` | Model receiving NumPy array instead of DataFrame | This is expected behaviour — safe to ignore |
| `ConnectionRefusedError` when calling the API | Flask server not running | Start the server with `python app.py` first |
