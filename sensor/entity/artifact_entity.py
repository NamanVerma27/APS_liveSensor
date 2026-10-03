from dataclasses import dataclass

@dataclass

class DataIngestionArtifact: # significance - holds file paths for training and testing datasets
    training_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact: # significance - holds file paths and status for data validation results
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataPreprocessingArtifact: # significance - holds file paths for preprocessed data and preprocessing object
    preprocessed_object_file_path: str
    processed_train_file_path: str
    processed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    cost: float = 0.0

@dataclass
class ModelTrainerArtifact: # significance - holds file path for the trained model
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact: # significance - holds evaluation status and report file path
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    trained_model_metric_artifact: ClassificationMetricArtifact
    challenger_model_metric_artifact: ClassificationMetricArtifact

@dataclass
class ModelPusherArtifact: # significance - holds file paths for the pushed model and its directory
    saved_model_path: str
    model_file_path: str