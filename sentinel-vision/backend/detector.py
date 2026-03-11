import cv2
import numpy as np
import time
from ultralytics import YOLO

class ThreatDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Define threat classes (COCO class ID mapping)
        # 0: person
        # 2-7: vehicles
        # 24: backpack, 28: suitcase
        # 43: knife
        self.THREAT_CLASSES = ['knife', 'gun']  # High priority threats
        self.SUSPICIOUS_OBJECTS = ['backpack', 'suitcase'] # Needs tracking
        self.VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']
        
        # Simplified tracking for abandoned objects
        # Format: {obj_id: {'class': name, 'first_seen': timestamp, 'last_seen': timestamp, 'location': (x, y)}}
        self.tracked_objects = {}
        
        # Define a mock restricted zone for intrusion detection (polygon points)
        # Normally this would be configurable via the UI
        self.restricted_zone = [(100, 100), (500, 100), (500, 400), (100, 400)]

    def is_in_restricted_zone(self, center_x, center_y):
        # A simple bounding box check for the restricted zone
        if self.restricted_zone:
            x_min = min(p[0] for p in self.restricted_zone)
            x_max = max(p[0] for p in self.restricted_zone)
            y_min = min(p[1] for p in self.restricted_zone)
            y_max = max(p[1] for p in self.restricted_zone)
            return x_min <= center_x <= x_max and y_min <= center_y <= y_max
        return False

    def process_frame(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        detected_threats = []
        persons_locations = []
        
        # Parse results
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf = float(box.conf[0])
            
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            threat_level = 'none'
            
            # Logic: Weapons
            if class_name in self.THREAT_CLASSES:
                threat_level = 'critical'
            
            # Logic: Intrusion
            elif class_name == 'person':
                persons_locations.append((center_x, center_y))
                if self.is_in_restricted_zone(center_x, center_y):
                    threat_level = 'high'
                    class_name = 'person (intrusion)'
            
            # Logic: Suspicious object (simplified)
            elif class_name in self.SUSPICIOUS_OBJECTS:
                threat_level = 'medium'
            
            # Track detected object
            if threat_level != 'none':
                detected_threats.append({
                    'object': class_name,
                    'level': threat_level,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
        
        return results.plot(), detected_threats, persons_locations
