import os , sys
import dill

import numpy as np
import yaml

from sensor.logger import logging as global_logging
from sensor.exception import SensorException

import logging
logger = logging.getLogger(__name__)

def read_yaml_file(file_path: str) -> dict:
    
    # Reads a YAML file and returns its contents as a dictionary

    try:
        with open(file_path , 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise SensorException(e , sys)
    
def write_yaml_file(file_path: str , content: object , replace: bool = False) -> None:
    # Writes a dictionary to a YAML file
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
                
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , 'w') as yaml_file:
            yaml.dump(content , yaml_file)
    
    except Exception as e:
        raise SensorException(e , sys)
    
def save_numpy_array_data(file_path: str , array: np.ndarray) -> None:
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save

    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        with open(file_path , 'wb') as file_obj:
            np.save(file_obj , array)
        
    except Exception as e:
        raise SensorException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """

    try:
        with open(file_path , 'rb') as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise SensorException(e , sys)
    
def save_object(file_path: str , obj: object) -> None:
    """
    save a python object to file
    file_path: str location of file to save
    obj: object to be saved
    """

    try:
        logger.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj , file_obj)
        logger.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise SensorException(e , sys)
    
def load_object(file_path: str) -> object:
    """
    load a python object from file
    file_path: str location of file to load
    return: object loaded
    """

    try:
        logger.info("Entered the load_object method of MainUtils class")
        with open(file_path , 'rb') as file_obj:
            obj = dill.load(file_obj)
        logger.info("Exited the load_object method of MainUtils class")
        return obj

    except Exception as e:
        raise SensorException(e , sys)