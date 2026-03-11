import cv2
import os
import time
import numpy as np
from detector import ThreatDetector
from face_tools import train_recognizer

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

class ArchiveProcessor:
    def __init__(self):
        self.detector = ThreatDetector()
        self.face_recognizer, self.label_map, _ = train_recognizer()
        
    def process_video(self, input_path, output_path, job_id, jobs_dict):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            jobs_dict[job_id]["status"] = "Failed"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Using 'avc1' (H.264) for better browser compatibility if available
        # Fallback to 'mp4v' if needed, but 'avc1' is preferred for web playback
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if out is valid, if not try mp4v as fallback
        if not out.isOpened():
            print("Warning: 'avc1' codec failed, falling back to 'mp4v'")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        findings = []
        last_seen = {} # object_name -> last_timestamp
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detector
            annotated_frame, threats, _ = self.detector.process_frame(frame)
            
            timestamp = frame_count / fps
            time_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

            # Run Face Recognizer
            if self.face_recognizer:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 6)
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.equalizeHist(face_roi)
                    label_id, confidence = self.face_recognizer.predict(face_roi)
                    
                    if confidence < 200: 
                        name = self.label_map.get(label_id, "Unknown")
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{name}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        obj_key = f"Watchlist: {name}"
                        if obj_key not in last_seen or (timestamp - last_seen[obj_key] > 2.0):
                            last_seen[obj_key] = timestamp
                            findings.append({
                                "id": f"{job_id}_{frame_count}_face",
                                "type": f"WATCHLIST: {name.upper()}",
                                "severity": "HIGH",
                                "timestamp": time_str,
                                "camera": "Archive"
                            })

            # Add YOLO threats to findings
            for t in threats:
                obj_name = t['object']
                if obj_name not in last_seen or (timestamp - last_seen[obj_name] > 2.0):
                    last_seen[obj_name] = timestamp
                    
                    # Map severity
                    severity = "MEDIUM"
                    if t['level'] == 'critical' or t['level'] == 'high':
                        severity = "HIGH"
                    elif t['level'] == 'low':
                        severity = "LOW"

                    findings.append({
                        "id": f"{job_id}_{frame_count}_{obj_name}",
                        "type": f"{obj_name.upper()} DETECTED",
                        "severity": severity,
                        "timestamp": time_str,
                        "camera": "Archive"
                    })

            out.write(annotated_frame)
            frame_count += 1
            
            # Update progress every 10%
            if frame_count % max(1, total_frames // 10) == 0:
                progress = int((frame_count / total_frames) * 100)
                jobs_dict[job_id]["progress"] = progress
                print(f"Job {job_id} progress: {progress}%")

        cap.release()
        out.release()
        
        # Final update
        jobs_dict[job_id]["status"] = "Completed"
        jobs_dict[job_id]["progress"] = 100
        jobs_dict[job_id]["findings"] = findings
        # Store relative URL for frontend - Using /api/uploads to match Vite proxy
        jobs_dict[job_id]["video_url"] = f"/api/uploads/{os.path.basename(output_path)}"
        print(f"Job {job_id} completed. Findings: {len(findings)}")

processor = ArchiveProcessor()
