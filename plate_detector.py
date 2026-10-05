"""License plate detector using YOLO model."""

from ultralytics import YOLO
import torch
from config import DEFAULT_PLATE_MODEL_PATH


class PlateDetector(object):
    """Detects license plates in images."""
    
    def __init__(self, model_path=DEFAULT_PLATE_MODEL_PATH, confidence_threshold=0.8,
                 input_size=(384, 384), use_gpu=True, imgsz=1280):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.imgsz = imgsz
        
        self.model = YOLO(model_path)
        if model_path.endswith('.pt'):
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self.model.to(device)

    def detect(self, image):
        """
        Detect license plates.
        
        Returns:
            List of dicts with 'box' [x1,y1,x2,y2] and 'score'
        """
        results = self.model(image, imgsz=self.imgsz, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            score = float(box.conf[0])
            if score < self.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections.append({'box': [x1, y1, x2, y2], 'score': score})
            
        return detections
