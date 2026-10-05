"""
Shared configuration constants for ANPR system
"""

import os

# Vehicle class IDs from custom dataset (kaggle.pt)
VEHICLE_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6]  
VEHICLE_CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "scooter",
    3: "autorickshaw",
    4: "truck",
    5: "bus",
    6: "e-rickshaw"
}

# Default YOLO model paths
DEFAULT_MODEL_PATH = "D:/FinalModels/kaggle.pt"
DEFAULT_PLATE_MODEL_PATH = "D:/FinalModels/ANPR_best.pt"

# License plate detection settings
MAX_PLATE_DETECTIONS_PER_VEHICLE = 3  # Max times to detect plate per vehicle
CROPPED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cropped")
