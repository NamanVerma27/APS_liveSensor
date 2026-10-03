import os,sys

from sensor.exception import SensorException
from sensor.logger import logging as global_logging

import logging
logger = logging.getLogger(__name__)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

from sensor.utils.main_utils import load_numpy_array_data
from sensor.utils.main_utils import save_object,load_object

from sensor.entity.artifact_entity import (
    DataPreprocessingArtifact,
    ModelTrainerArtifact)
from sensor.entity.config_entity import ModelTrainerConfig

from sensor.ml_model_components.metric.classification_metric import get_classification_score
from sensor.ml_model_components.model.estimator import SensorModel

class ModelTrainer:
    def __init__(self ,data_preprocessing_artifact : DataPreprocessingArtifact,
                  model_trainer_config : ModelTrainerConfig ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_preprocessing_artifact
        except Exception as e:
            raise SensorException(e,sys)
    
    def perform_hyper_parameter_tuning(self , x_train, y_train):
        """
        Logic: Tries various combinations of hyperparameters to find the 
        best settings for the XGBoost model.
        """
        try:
            logger.info("Starting Hyperparameter tuning")
            # Define the parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.1, 0.01, 0.05]
            }

            xgb_clf = XGBClassifier()
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=xgb_clf,
                param_grid=param_grid,
                cv=3, # 3-fold cross-validation
                verbose=1,
                scoring='f1' # We prioritize F1-score for imbalanced sensor data
            )

            grid_search.fit(x_train, y_train)
            
            logger.info(f"Best Parameters found: {grid_search.best_params_}")
            return grid_search.best_params_

        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x_train, y_train, best_params=None, model_type="xgboost"):
        """
        Trains either XGBoost or CatBoost using tuned or default parameters.
        """
        try:
            if model_type.lower() == "catboost":
                params = best_params if best_params is not None else {"verbose": False, "random_state": 42}
                model = CatBoostClassifier(**params)
            else:
                params = best_params if best_params is not None else {"random_state": 42, "eval_metric": "logloss"}
                model = XGBClassifier(**params)
            
            model.fit(x_train, y_train)
            return model
        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # ------1. Load transformed training and testing arrays------
            logger.info("Loading transformed training and testing arrays")

            train_file_path = self.data_transformation_artifact.processed_train_file_path
            test_file_path = self.data_transformation_artifact.processed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            #------2. Split training and testing arrays into input and target feature------
            logger.info("Splitting training and testing arrays into input and target feature")
            x_train , y_train = train_arr[:,:-1] , train_arr[:,-1]
            x_test , y_test = test_arr[:,:-1] , test_arr[:,-1]

            #------3. Train the model------
            # First, find the best parameters
            best_params = self.perform_hyper_parameter_tuning(x_train, y_train)

            # Then, train the final model with those parameters
            logger.info("Training the final model with best hyperparameters")
            model = self.train_model(x_train , y_train, best_params=best_params)

            #------4. Calculate training and testing accuracy------
            logger.info("Calculating training and testing accuracy")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_metric = get_classification_score(y_true=y_train , y_pred=y_train_pred)
            test_metric = get_classification_score(y_true=y_test , y_pred=y_test_pred)
            logger.info(f"Training metric : {train_metric}")
            logger.info(f"Testing metric : {test_metric}")

            #------5. Check expected score floor and overfitting/underfitting------
            if test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise SensorException(
                    f"Model test score ({test_metric.f1_score:.4f}) is below expected accuracy threshold ({self.model_trainer_config.expected_accuracy:.4f})",
                    sys
                )

            logger.info("Checking for overfitting and underfitting")
            diff = abs(train_metric.f1_score - test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise SensorException("Model is overfitting or underfitting. Try to improve the model.",sys)
            
            #------6. Save the trained model------
            logger.info("Saving the trained model")
            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            sensor_model = SensorModel(preprocessor=preprocessor , model=model)
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            save_object(file_path=trained_model_file_path , obj=sensor_model)
            logger.info(f"Trained model saved at : {trained_model_file_path}")

            #------7. Prepare the ModelTrainerArtifact------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            logger.info(f"Model Trainer Artifact : {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise SensorException(e,sys)