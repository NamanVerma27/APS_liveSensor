# 🚛 APS_sensorlive: Industrial Air Pressure System (APS) Fault Detection

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1%2B-EB5424.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow.svg)](https://catboost.ai/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg?logo=mongodb)](https://www.mongodb.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning system engineered for predictive maintenance of **Scania Heavy-Duty Trucks**. The system predicts pneumatic **Air Pressure System (APS)** component failures, optimizing maintenance scheduling to minimize operational downtime and prevent catastrophic highway breakdowns.

---

## 📌 Executive Summary & Business Formulation

The Air Pressure System (APS) in commercial vehicles provides compressed air for safety-critical operations, primarily braking and gear-changing subsystems. Unpredicted APS failures lead to roadside breakdowns, towing costs, and hazardous conditions.

### The Asymmetric Cost Matrix
Traditional classification models optimize for symmetric accuracy or F1-score, which fails in high-stakes predictive maintenance. This project is directly aligned with the **Scania APS Failure Cost Function**:

| Outcome | Technical Term | Operational Impact | Cost (USD) |
| :--- | :--- | :--- | :---: |
| **False Positive (FP)** | Type I Error | Truck inspected by mechanic; no fault found | **$10** |
| **False Negative (FN)** | Type II Error | Truck misses inspection; breakdown on road | **$500** |
| **True Positive (TP)** | Correct Catch | Necessary preventative maintenance performed | Negligible |
| **True Negative (TN)** | Normal Status | Healthy truck continues commercial operation | $0 |

$$\text{Total Operational Cost} = (10 \times \text{False Positives}) + (500 \times \text{False Negatives})$$

> **Key Takeaway:** A missed failure is **50 times more expensive** than a false alarm. The entire data engineering and modeling pipeline is calibrated to minimize this cost metric.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A["MongoDB Atlas (Cloud Feature Store)"] --> B["Data Ingestion Engine"]
        B -->|Train (80%)| C1["Train Dataset"]
        B -->|Test (20%)| C2["Holdout Test Dataset"]
    end

    subgraph Validation & Hygiene
        C1 & C2 --> D["Data Validation Component"]
        D -->|KS Drift Test (p > 0.05)| E["Validated Datasets"]
        D -->|Drift Detected (p <= 0.05)| D_FAIL["Pipeline Early Exit & Alert"]
    end

    subgraph Feature Engineering
        E --> F["Data Preprocessing Pipeline"]
        F --> F1["SimpleImputer (constant fill=0)"]
        F1 --> F2["RobustScaler (outlier suppression)"]
        F2 --> F3["SMOTETomek (Train set only)"]
        F3 --> G["Transformed NumPy Tensors (.npy)"]
    end

    subgraph Model Arena & Evaluation
        G --> H["Model Trainer (GridSearchCV Tuning)"]
        H -->|Champion Bundle (SensorModel)| I["Challenger Model"]
        I --> J["Model Evaluation (Holdout Test Only)"]
        K["Production Champion (saved_models/)"] --> J
        J -->|Challenger Beats Champion by > 0.02| L["Model Pusher Component"]
        J -->|Fails Threshold| M["Model Rejected & Audited"]
    end

    subgraph Production Deployment
        L --> N["Timestamped Model Registry (saved_models/<timestamp>/)"]
        N --> O["Prediction Pipeline (Feature Alignment Engine)"]
        O --> P["FastAPI Microservice (REST API)"]
        P -->|BackgroundTasks| Q["Async /train Route"]
        P -->|CSV Streaming| R["Batch /predict Route"]
        P -->|JSON Telemetry| S["Real-Time /predict-live Route"]
    end
```

---

## 🔬 Exploratory Data Analysis & Scientific Learnings

1. **High Dimensionality & Sparse Sensors:**
   The raw dataset contains **171 columns** (1 target + 170 anonymized sensor metrics). 7 features (`br_000`, `bq_000`, `bp_000`, `ab_000`, `cr_000`, `bo_000`, `bn_000`) exhibited **$> 70\%$ missing values** and were eliminated to remove noise.
2. **Extreme Positive Skewness:**
   Sensors such as `aa_000` and `co_000` exhibit extreme right-skewness (medians near 8 vs means exceeding 300,000). A `RobustScaler` was selected over standard normalization to eliminate distortion from extreme sensor spikes.
3. **Severe Class Imbalance:**
   The target distribution contains **~2.7% positive failure cases** (`pos`). Class balance is treated using `SMOTETomek` strictly on transformed training data, preventing information leakage into the test set.

### Model Benchmark & Bake-Off Comparison

Benchmarking conducted on identical holdout sets evaluated against the exact business cost metric:

| Rank | Model Architecture | False Negatives (FN) | False Positives (FP) | Total Cost (USD) | Benchmark Time |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **`CatBoostClassifier`** | **33** | 52 | **$17,020** | ~77 s |
| 🥈 | **`XGBClassifier`** | 37 | 44 | **$18,940** | ~6 s |
| 🥉 | **`RandomForestClassifier`** | 37 | 71 | **$19,210** | ~25 s |
| 4 | **`LogisticRegression`** | 10 | 4,652 | **$51,520** | ~45 s |

> **Business Win:** CatBoost caught **4 additional failing trucks** compared to XGBoost, yielding a net savings of **$1,920** per maintenance cohort.

---

## 📂 Project Directory Structure

```
APS_sensorlive/
├── .env.example                       # Environment variable template
├── .gitignore                         # Enterprise gitignore patterns
├── config/
│   └── schema.yaml                    # Dataset schema & drop-column definitions
├── data/                              # Raw sensor datasets
│   └── aps_failure_training_set1.csv  # Scania industrial failure dataset
├── logs/                              # Timestamped rolling application logs
├── main.py                            # FastAPI application & REST routing
├── notebooks/                         # Research & exploration notebooks
│   ├── EDA.ipynb                      # Complete exploratory data analysis
│   ├── dataPreprocessing.ipynb        # Pipeline optimization & model bake-off
│   └── ElasticNet.ipynb               # Linear regularized comparison
├── requirements.txt                   # Production dependency manifest (UTF-8)
├── setup.py                           # Python package installation configuration
└── sensor/                            # Core industrial package
    ├── __init__.py                    # Automatic .env loading initialization
    ├── logger.py                      # Root-anchored structured logging
    ├── exception.py                   # Safe traceback parsing & diagnostic formatting
    ├── configuration/                 # Infrastructure clients
    │   └── mongo_db_connection.py     # Thread-safe MongoDB client singleton
    ├── constant/                      # Central constants & threshold parameters
    │   ├── application.py             # Server host, port definitions
    │   ├── database.py                # Database and collection identifiers
    │   ├── env_variable.py            # Environment variable key names
    │   └── training_pipeline/         # Pipeline paths, ratios, and thresholds
    ├── data_access/                   # Data layer abstraction
    │   └── sensor_data.py             # MongoDB DataFrame I/O & batch ingestion
    ├── entity/                        # Data contracts
    │   ├── config_entity.py           # Stage configurations with instance try-catch
    │   └── artifact_entity.py         # Stage metadata artifacts
    ├── ml_model_components/           # Reusable machine learning components
    │   ├── metric/
    │   │   └── classification_metric.py # Precision, Recall, F1, and Cost metrics
    │   └── model/
    │       ├── estimator.py           # SensorModel bundle (preprocessor + model)
    │       └── model_resolver.py      # Production model registry resolver
    ├── components/                    # Core pipeline stages
    │   ├── data_ingestion.py          # MongoDB extraction & train/test split
    │   ├── data_validation.py         # Schema validation & KS drift detection
    │   ├── data_preprocessing.py      # Transformation, scaling & SMOTETomek
    │   ├── model_trainer.py           # GridSearchCV hyperparameter optimization
    │   ├── model_evaluation.py        # Independent holdout comparison (leak-free)
    │   └── model_pusher.py            # Automated deployment to registry
    ├── pipeline/                      # Workflow orchestrators
    │   ├── training_pipeline.py       # End-to-end training pipeline with guards
    │   └── prediction_pipeline.py     # Schema-aligned inference engine
    └── utils/
        └── main_utils.py              # YAML, NumPy, and dill serialization
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
* Python 3.10 or 3.11
* MongoDB Atlas cluster or local MongoDB instance

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/NamanVerma27/APS_liveSensor.git
cd APS_liveSensor

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Upgrade pip and install package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Credentials
Copy the `.env.example` file to `.env` and configure your MongoDB connection string:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
MONGO_DB_URL="mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
```

---

## ⚡ Running the API & Inference

### 1. Start the FastAPI Service
```bash
python main.py
```
The server will bind to `http://0.0.0.0:8080`.
* **Interactive Documentation (Swagger):** `http://localhost:8080/docs`
* **Alternative Documentation (ReDoc):** `http://localhost:8080/redoc`

### 2. Trigger End-to-End Pipeline Training
The training route operates asynchronously in the background, keeping the API responsive for predictions:
```bash
curl -X GET "http://localhost:8080/train"
```
**Response:**
```json
{
  "status": "success",
  "message": "Training pipeline initiated in background."
}
```

### 3. Batch Inference (CSV Upload)
Predict on a batch CSV file:
```bash
curl -X POST "http://localhost:8080/predict" \
     -F "file=@test_sensors.csv"
```
**JSON Response:**
```json
{
  "status": "success",
  "total_records": 1000,
  "predicted_pos_failures": 28,
  "predicted_neg_normal": 972,
  "predictions": [
    { "aa_000": 76698, "prediction": "pos" },
    { "aa_000": 33058, "prediction": "neg" }
  ]
}
```

To stream and download the full predictions as a CSV file:
```bash
curl -X POST "http://localhost:8080/predict?download=true" \
     -F "file=@test_sensors.csv" \
     --output predictions.csv
```

### 4. Real-Time Telemetry Inference (Single Reading)
Send JSON telemetry directly from an onboard IoT sensor device:
```bash
curl -X POST "http://localhost:8080/predict-live" \
     -H "Content-Type: application/json" \
     -d '{"aa_000": 76698, "ac_000": 2130706432, "ad_000": 366, "ae_000": 0}'
```
**Response:**
```json
{
  "status": "success",
  "prediction": "pos",
  "interpretation": "APS Failure (Urgent Maintenance)"
}
```

---

## 🛡️ Engineering Best Practices & Quality Controls

* **Zero Data Leakage:** Preprocessors and scalers are fitted solely on training splits. Holdout evaluations are conducted strictly on independent test splits.
* **Model Bundling (`SensorModel`):** The preprocessor pipeline and classifier model are serialized as a single artifact, eliminating train-serving skew.
* **Schema Alignment Engine:** The prediction pipeline dynamically inspects `sensor_model.preprocessor.feature_names_in_`, gracefully tolerating missing or reordered input columns.
* **Drift Detection Guard:** Employs the two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`). Pipeline halts gracefully with diagnostic report generation if distribution drift exceeds $\alpha = 0.05$.
* **Robust Exception Safety:** Custom `SensorException` safely extracts execution frames without risking secondary `NoneType` attribute errors.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author & Acknowledgments

* **Lead Engineer:** Naman Verma ([@NamanVerma27](https://github.com/NamanVerma27))
* **Dataset:** Scania CV AB (Industrial challenge dataset for Air Pressure System component failure prediction)
