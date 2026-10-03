import os
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    requirements_list: List[str] = []
    req_file_path = "requirements.txt"
    if os.path.exists(req_file_path):
        with open(req_file_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-e"):
                    requirements_list.append(line)
    return requirements_list

setup(
    name="APS_sensorlive",
    version="0.1.0",
    author="Naman",
    author_email="namanv417@gmail.com",
    description="Air Pressure System (APS) Sensor Fault Prediction for Scania Heavy Trucks",
    packages=find_packages(),
    install_requires=get_requirements()
)