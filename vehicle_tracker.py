"""Vehicle detection and tracking using YOLO + ByteTrack."""

import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO
import torch

from config import DEFAULT_MODEL_PATH, VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES


class VehicleTracker(object):
    """Detects and tracks vehicles using ByteTrack."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.8,
                 input_size=(384, 384), use_gpu=True, track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Load YOLO model
        self.model = YOLO(model_path)
        if model_path.endswith('.pt'):
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self.model.to(device)
        self.model_names = self.model.names
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        self.track_history = defaultdict(list)
        self.max_history = 50

    def detect(self, image):
        """Detect vehicles, returns supervision.Detections."""
        results = self.model(image, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Determine valid vehicle class IDs for this model
        valid_classes = []
        for cid in class_ids:
            name = self.model_names.get(cid, '').lower()
            if not self.model_names or len(self.model_names) <= 2:
                # Custom/dedicated vehicle detector model (very few classes)
                valid_classes.append(cid)
            elif name in ['car', 'motorcycle', 'bus', 'truck', 'vehicle', 'van', 'suv', 'auto']:
                valid_classes.append(cid)
            elif cid in VEHICLE_CLASS_IDS:
                valid_classes.append(cid)
                
        valid_classes = list(set(valid_classes))
        
        # Filter detections
        mask = (confs >= self.confidence_threshold)
        if valid_classes:
            mask = mask & np.isin(class_ids, valid_classes)
            
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
        
        if len(boxes) == 0:
            return sv.Detections.empty()
            
        return sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=confs.astype(np.float32),
            class_id=class_ids.astype(int)
        )

    def track(self, image):
        """Detect and track vehicles."""
        detections = self.detect(image)
        if len(detections) == 0:
            return detections
        
        tracked = self.tracker.update_with_detections(detections)
        
        # Update trajectory history
        if tracked.tracker_id is not None:
            for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                history = self.track_history[tid]
                history.append(center)
                if len(history) > self.max_history:
                    history.pop(0)
        
        return tracked

    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.track_history.clear()

    def set_frame_rate(self, fps):
        """Reinitialize tracker with new frame rate."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=fps
        )

    @staticmethod
    def get_class_name(class_id):
        if class_id in VEHICLE_CLASS_NAMES:
            return VEHICLE_CLASS_NAMES[class_id]
        coco_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck"}
        return coco_names.get(class_id, f"vehicle_{class_id}")
