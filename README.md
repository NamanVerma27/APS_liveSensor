# APS_sensorlive: Scania Truck Air Pressure System (APS) Fault Detection

An end-to-end Machine Learning system for predictive maintenance of Scania heavy-duty truck Air Pressure Systems (APS). The system classifies whether component breakdowns are related to the APS sensor sub-system, minimizing false alarms ($10 cost) and catastrophic unpredicted breakdowns ($500 cost).

## Architecture & Pipeline Flow

The project follows a component-based MLOps architecture:
1. **Data Ingestion:** Extracts sensor data from MongoDB Atlas, performs feature pruning (>70% null columns), and stratifies into train/test splits.
2. **Data Validation:** Schema conformance verification and Kolmogorov-Smirnov distribution drift testing.
3. **Data Preprocessing:** Imputation (`fill_value=0`), `RobustScaler`, and `SMOTETomek` minority oversampling on training data.
4. **Model Training:** Hyperparameter optimization (`GridSearchCV`) and training of tree-based ensembles (XGBoost / CatBoost).
5. **Model Evaluation:** Independent holdout test evaluation comparing challenger against the production champion model.
6. **Model Pusher:** Automated deployment of accepted models into the timestamped production model registry.
7. **FastAPI Serving:** Non-blocking async endpoints for model training (`/train`) and batch CSV prediction (`/predict`).

## Project Setup

### 1. Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Environment Variables
Create a `.env` file based on `.env.example`:
```env
MONGO_DB_URL=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
```

### 3. Run FastAPI Web Service
```bash
python main.py
```
Access the interactive Swagger documentation at `http://localhost:8080/docs`.

### 4. API Endpoints
* `GET /docs`: Interactive Swagger UI.
* `GET /train`: Triggers the end-to-end training pipeline asynchronously in the background.
* `POST /predict`: Upload a CSV file of sensor readings to receive predictions or download an annotated results file.
