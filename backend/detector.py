import cv2
import numpy as np
import time
from ultralytics import YOLO

class ThreatDetector:
    def __init__(self, model_path='yolo26n.pt'):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.rules = [] # List of active RuleDB objects
        
        # Keep internal restricted zone for intrusion fallback
        self.restricted_zone = [(100, 100), (500, 100), (500, 400), (100, 400)]

    def set_rules(self, rules):
        """Update the active rules for this detector instance."""
        self.rules = rules

    def is_in_restricted_zone(self, center_x, center_y):
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
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Parse results
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf = float(box.conf[0])
            
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            if class_name == 'person':
                persons_locations.append((center_x, center_y))

            # Logic check: Rules vs Defaults
            matching_rule = None
            for rule in self.rules:
                # Find if any rule (enabled or not) targets this object
                targets = [t.strip().lower() for t in rule.target.split(',')]
                is_intrusion = rule.category in ['Zone', 'Behavior'] and 'intrusion' in rule.target.lower()
                
                if class_name.lower() in targets or (is_intrusion and class_name == 'person' and self.is_in_restricted_zone(center_x, center_y)):
                    matching_rule = rule
                    break
            
            final_threat = None
            
            # Case 1: Matching Rule Found
            if matching_rule:
                if matching_rule.enabled:
                    # Show with rule-specific level/label
                    display_name = "Intrusion" if 'intrusion' in matching_rule.target.lower() else class_name
                    final_threat = {
                        'object': display_name,
                        'level': matching_rule.alert_severity.lower(),
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    }
                else:
                    # Rule exists but is disabled -> HIDE (suppress)
                    continue
            
            # Case 2: No Rule -> Show by default (Normal Behavior)
            else:
                final_threat = {
                    'object': class_name,
                    'level': 'info', # Default visibility level
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                }
            
            # Draw the box if it hasn't been suppressed
            if final_threat:
                # Add to alerts if it's significant (rule match or default person/weapon)
                # For 'info' level, we might not want to log it in the DB, 
                # but we'll show it in the feed.
                if final_threat['level'] != 'info' or class_name.lower() in ['person', 'weapon', 'gun', 'knife', 'firearm']:
                    detected_threats.append(final_threat)
                
                # Styling
                if final_threat['level'] == 'high': color = (0, 0, 255) # Red
                elif final_threat['level'] == 'medium': color = (0, 165, 255) # Orange
                elif final_threat['level'] == 'low': color = (0, 255, 0) # Green
                else: color = (200, 200, 200) # Gray for 'info'
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{final_threat['object']} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated_frame, detected_threats, persons_locations
