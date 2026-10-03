import io
import os
import sys
import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.responses import Response
from starlette.responses import RedirectResponse, StreamingResponse

from sensor.constant.application import APP_HOST, APP_PORT
from sensor.exception import SensorException
from sensor.logger import logging, logger
from sensor.pipeline.prediction_pipeline import PredictionPipeline
from sensor.pipeline.training_pipeline import TrainPipeline

app = FastAPI(
    title="APS_sensorlive API",
    description="Air Pressure System (APS) Fault Detection REST API for Scania Heavy Trucks",
    version="0.1.0"
)


@app.get("/", tags=["documentation"])
async def root():
    """Redirects the root URL to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def train_route(background_tasks: BackgroundTasks):
    """
    Non-blocking endpoint to trigger the end-to-end training pipeline.
    Executes in the background via BackgroundTasks to keep the server responsive.
    """
    try:
        if TrainPipeline.is_pipeline_running:
            return Response("Training pipeline is already running in background.")

        train_pipeline = TrainPipeline()
        background_tasks.add_task(train_pipeline.run_pipeline)
        return {"status": "success", "message": "Training pipeline initiated in background."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(SensorException(e, sys)))


@app.post("/predict", tags=["prediction"])
async def predict_route(file: UploadFile = File(...), download: bool = False):
    """
    Batch prediction endpoint. Accepts a CSV file of sensor readings.
    If download=true, returns a downloadable CSV stream with predictions.
    If download=false, returns full JSON prediction results and failure counts.
    """
    try:
        df = pd.read_csv(file.file)
        pred_pipeline = PredictionPipeline()
        prediction_df = pred_pipeline.predict(df)

        if download:
            stream = io.StringIO()
            prediction_df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
            return response

        return {
            "status": "success",
            "total_records": len(prediction_df),
            "predicted_pos_failures": int((prediction_df["prediction"] == "pos").sum()),
            "predicted_neg_normal": int((prediction_df["prediction"] == "neg").sum()),
            "predictions": prediction_df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-live", tags=["prediction"])
async def predict_live_route(data: dict):
    """
    Real-time single-record prediction endpoint for streaming truck telemetry.
    Accepts sensor readings as a JSON key-value dictionary.
    """
    try:
        df = pd.DataFrame([data])
        pred_pipeline = PredictionPipeline()
        prediction_df = pred_pipeline.predict(df)
        prediction_value = str(prediction_df["prediction"].iloc[0])
        return {
            "status": "success",
            "prediction": prediction_value,
            "interpretation": "APS Failure (Urgent Maintenance)" if prediction_value == "pos" else "Normal (No APS Fault)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)