import os , sys
import pandas as pd

from sensor.logger import logging as global_logging
from sensor.exception import SensorException

import logging
logger = logging.getLogger(__name__)

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.config_entity import ModelEvaluationConfig
from dataclasses import asdict
from sensor.entity.artifact_entity import ( DataValidationArtifact,
                                           ModelTrainerArtifact,
                                           ModelEvaluationArtifact)

from sensor.utils.main_utils import save_object,load_object,write_yaml_file
from sensor.ml_model_components.metric.classification_metric import get_classification_score
from sensor.ml_model_components.model.estimator import ( SensorModel,
                                                        TargetValueMapping )
from sensor.ml_model_components.model.model_resolver import ModelResolver

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation stage")
            
            # 1. Load and Prepare the Holdout Evaluation Arena (Test Set Only to prevent leakage)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            y_true = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
            x_eval = test_df.drop(TARGET_COLUMN, axis=1)

            # 2. Identify the Challenger (The model we just trained)
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True  # Default assumption

            # 3. Handle the "First Run" Scenario
            if not model_resolver.is_model_exists():
                logger.info("No production model found. Automatically accepting the current trained model.")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=0.0,
                    best_model_path=None,
                    trained_model_path=trained_model_path,
                    trained_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    challenger_model_metric_artifact=None
                )
                return model_evaluation_artifact

            # 4. Handle the "Battle" Scenario (Production model exists)
            logger.info("Production model exists. Loading models for comparison.")
            best_model_path = model_resolver.get_best_model_path()
            
            # Load the Champion and the Challenger
            best_model = load_object(file_path=best_model_path)
            trained_model = load_object(file_path=trained_model_path)

            # Generate Predictions using the full evaluation set
            y_trained_pred = trained_model.predict(x_eval)
            y_best_pred = best_model.predict(x_eval)

            # Calculate metrics
            trained_metric = get_classification_score(y_true, y_trained_pred)
            best_metric = get_classification_score(y_true, y_best_pred)

            # 5. The Decision Matrix
            improved_accuracy = trained_metric.f1_score - best_metric.f1_score
            
            if improved_accuracy > self.model_eval_config.change_threshold:
                is_model_accepted = True
                logger.info(f"SUCCESS: New model performance ({trained_metric.f1_score}) "
                             f"beats production ({best_metric.f1_score}) by {improved_accuracy:.4f}")
            else:
                is_model_accepted = False
                logger.info(f"REJECTED: New model improvement ({improved_accuracy:.4f}) "
                             f"is below threshold ({self.model_eval_config.change_threshold})")

            # 6. Final Artifact & Reporting
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=best_model_path, # Path to the Champion model - the previous best
                trained_model_path=trained_model_path, # Path to the Challenger model - the newly trained
                trained_model_metric_artifact=trained_metric,
                challenger_model_metric_artifact=best_metric
            )

            # Save results to YAML for audit trail
            model_eval_report = asdict(model_evaluation_artifact)
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)

            logger.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)