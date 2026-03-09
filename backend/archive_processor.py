import cv2
import os
from detector import ThreatDetector

def process_video_file(input_path, output_path):
    print(f"Starting analysis on {input_path}")
    detector = ThreatDetector()
    detector.restricted_zone = [] # Clear bounds
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or fps != fps:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    findings = []
    frame_count = 0
    last_seen = {}
    debounced_findings = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, threats, _ = detector.process_frame(frame)
        out.write(annotated_frame)
        
        timestamp_sec = frame_count / fps
        
        for t in threats:
            obj_class = t['object']
            # Only record a finding if we haven't seen this object type in the last 2 seconds
            if obj_class not in last_seen or (timestamp_sec - last_seen[obj_class]) >= 2.0:
                debounced_findings.append({
                    "id": f"event_{obj_class}_{int(timestamp_sec)}_{frame_count}",
                    "type": f"{obj_class.upper()} DETECTED",
                    "severity": "HIGH" if t['level'] in ['critical', 'high'] else "MEDIUM",
                    "timestamp": f"00:{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}",
                    "details": f"Confidence: {t['confidence']*100:.1f}%",
                    "camera": "Archive Upload"
                })
                last_seen[obj_class] = timestamp_sec
                
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Finished analysis. Found {len(debounced_findings)} unique events.")
    return debounced_findings
