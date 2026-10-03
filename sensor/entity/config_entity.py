from datetime import datetime
import os, sys

from sensor.constant.training_pipeline import (
    PIPELINE_NAME, ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, FILE_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR, DATA_INGESTION_INGESTED_DIR,
    TEST_FILE_NAME, DATA_INGESTION_COLLECTION_NAME, TRAIN_FILE_NAME,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
)
from sensor.constant.training_pipeline import (
    DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_VALID_DIR,
    DATA_VALIDATION_INVALID_DIR, DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
)
from sensor.constant.training_pipeline import (
    DATA_PREPROCESSING_DIR_NAME, DATA_PREPROCESSING_PROCESSED_DATA_DIR,
    DATA_PREPROCESSING_PROCESSED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME
)
from sensor.constant.training_pipeline import (
    MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME, MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
)
from sensor.constant.training_pipeline import (
    MODEL_EVALUATION_DIR_NAME, MODEL_EVALUATION_REPORT_NAME,
    MODEL_EVALUATION_THRESHOLD_SCORE
)
from sensor.constant.training_pipeline import (
    MODEL_PUSHER_DIR_NAME, MODEL_PUSHER_SAVED_MODEL_DIR, MODEL_FILE_NAME
)

from sensor.exception import SensorException


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.timestamp = datetime.now().strftime("%m-%d-%Y__%H-%M-%S")
            self.pipeline_name = PIPELINE_NAME
            self.artifact_dir = os.path.join(ARTIFACT_DIR, self.timestamp)
        except Exception as e:
            raise SensorException(e, sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
            self.test_size = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
            self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        except Exception as e:
            raise SensorException(e, sys)


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
            self.valid_data_dir = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
            self.invalid_data_dir = os.path.join(self.data_validation_dir, DATA_VALIDATION_INVALID_DIR)
            self.valid_train_file_path = os.path.join(self.valid_data_dir, TRAIN_FILE_NAME)
            self.valid_test_file_path = os.path.join(self.valid_data_dir, TEST_FILE_NAME)
            self.invalid_train_file_path = os.path.join(self.invalid_data_dir, TRAIN_FILE_NAME)
            self.invalid_test_file_path = os.path.join(self.invalid_data_dir, TEST_FILE_NAME)
            
            self.drift_report_dir = os.path.join(self.data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR)
            self.drift_report_file_path = os.path.join(self.drift_report_dir, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


class DataPreprocessingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_preprocessing_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)
            self.processed_data_dir = os.path.join(self.data_preprocessing_dir, DATA_PREPROCESSING_PROCESSED_DATA_DIR)
            self.processed_train_file_path = os.path.join(self.processed_data_dir, TRAIN_FILE_NAME.replace(".csv", ".npy"))
            self.processed_test_file_path = os.path.join(self.processed_data_dir, TEST_FILE_NAME.replace(".csv", ".npy"))
            self.preprocessed_object_dir = os.path.join(self.data_preprocessing_dir, DATA_PREPROCESSING_PROCESSED_OBJECT_DIR)
            self.preprocessed_object_file_path = os.path.join(self.preprocessed_object_dir, PREPROCESSING_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
            self.trained_model_dir = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR)
            self.trained_model_file_path = os.path.join(self.trained_model_dir, MODEL_TRAINER_TRAINED_MODEL_NAME)
            self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
            self.overfitting_underfitting_threshold = MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        except Exception as e:
            raise SensorException(e, sys)


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_evaluation_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
            self.report_file_path = os.path.join(self.model_evaluation_dir, MODEL_EVALUATION_REPORT_NAME)
            self.change_threshold = MODEL_EVALUATION_THRESHOLD_SCORE
        except Exception as e:
            raise SensorException(e, sys)


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            # 1. Location in ARTIFACT folder (History)
            self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME)
            self.model_file_path = os.path.join(self.model_pusher_dir, MODEL_FILE_NAME)

            # 2. Location in PRODUCTION folder (For the Resolver)
            timestamp = round(datetime.now().timestamp())
            self.saved_model_path = os.path.join(
                MODEL_PUSHER_SAVED_MODEL_DIR, 
                f"{timestamp}", 
                MODEL_FILE_NAME
            )
        except Exception as e:
            raise SensorException(e, sys)