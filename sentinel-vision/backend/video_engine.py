import cv2
import threading
import time
from detector import ThreatDetector
from database import SessionLocal
from models import EventDB
from datetime import datetime
import numpy as np
import asyncio
from face_tools import train_recognizer, face_cascade

class VideoEngine:
    def __init__(self, source=0):
        self.source = source
        print("Initializing YOLO ThreatDetector. This may download weights...")
        self.detector = ThreatDetector()
        
        print("Training Face Recognition AI...")
        self.face_recognizer, self.label_map, self.thumbnail_map = train_recognizer()
        self.thumbnail_images = {} # Cache for loaded cv2 images
        
        self.current_frame = None
        self.heatmap_frame = None
        self.latest_alerts = []
        self.running = False
        self.heatmap_data = [] 
        self.current_fps = 0.0
        self._frame_times = []
        self.lock = threading.Lock()
        self.rules = []

    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def stop(self):
        self.running = False

    def set_rules(self, rules):
        with self.lock:
            self.rules = rules

    def update_face_recognizer(self):
        print(f"Re-training Face Recognition AI for {self.source}...")
        results = train_recognizer()
        new_recognizer, new_label_map, new_thumbnail_map = results
        with self.lock:
            self.face_recognizer = new_recognizer
            self.label_map = new_label_map
            self.thumbnail_map = new_thumbnail_map
            self.thumbnail_images = {} # Clear cache on re-train
        print(f"Re-training complete for {self.source}. New Label Map: {new_label_map}")

    def _update(self):
        print("Starting video engine background thread...")
        
        # Load detector FIRST before locking the camera so downloads don't hold the lock
        db = SessionLocal()
        last_alert_time = {}
        
        # Initialize video
        if isinstance(self.source, int) or str(self.source).isdigit():
            print(f"Opening camera {self.source} with DSHOW backend...")
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            print(f"Opening video source {self.source}...")
            cap = cv2.VideoCapture(self.source)
            
        if cap.isOpened():
            print(f"Successfully connected to camera: {self.source}")
        else:
            print(f"CRITICAL: Failed to connect to camera: {self.source}")
            if isinstance(self.source, str) and "." in self.source:
                print(f"TIP: If {self.source} is an IP camera, ensure you use the full RTSP URL (e.g., rtsp://user:pass@IP/path)")
        
        try:
            while self.running:
                if not cap.isOpened():
                    print(f"Camera {self.source} not open, retrying in 5 seconds...")
                    time.sleep(5)
                    if isinstance(self.source, int) or str(self.source).isdigit():
                        cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(self.source)
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera")
                    if isinstance(self.source, str) and not self.source.startswith('rtsp'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(1)
                    continue
                
                # print(f"DEBUG: Frame grabbed from {self.source}") # Too noisy, but good for local check
                
                # Process threats
                try:
                    annotated_frame, threats, persons = self.detector.process_frame(frame)
                except Exception as e:
                    print(f"Error during YOLO processing: {e}")
                    continue
                
                # Apply Local Facial Recognition
                with self.lock:
                    current_recognizer = self.face_recognizer
                    current_label_map = self.label_map
                    current_thumbnail_map = self.thumbnail_map
                    current_rules = list(self.rules)

                if current_recognizer is not None:
                    try:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
                        
                        for (x, y, w, h) in faces:
                            face_roi = gray_frame[y:y+h, x:x+w]
                            # Normalize lighting
                            face_roi = cv2.equalizeHist(face_roi)
                            label_id, confidence = current_recognizer.predict(face_roi)
                            
                            # Confidence lower is better for LBPH (distance). < 80 is generally a good match.
                            # Threshold tightened to 200 to ensure high-confidence matches only
                            if confidence < 200:
                                name = current_label_map.get(label_id, "Unknown").replace("_", " ")
                                print(f"IDENTIFIED: {name} (Conf: {int(confidence)})")
                                text = f"{name} ({int(confidence)})"
                                color = (0, 255, 0) # Green for known

                                # Add face detection to threats - Set level to 'High' for watchlist matches
                                threats.append({'object': f'Watchlist: {name}', 'level': 'High', 'confidence': (100-max(0, confidence-150))/100, 'center': (x+w//2, y+h//2)})
                            else:
                                name = "Unknown"
                                print(f"UNKNOWN face detected (Conf: {int(confidence)})")
                                text = name
                                color = (0, 0, 255) # Red for unknown
                                
                            # Draw face bounding box and name on the frame
                            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(annotated_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        print(f"Error during Face Recognition: {e}")
                
                self.heatmap_data.extend(persons)
                self.heatmap_data = self.heatmap_data[-2000:] # Cap heatmap data size

                new_alerts = []
                current_time = time.time()
                
                # Filter threats by rules if rules are active
                final_threats = []
                if current_rules:
                    enabled_rules = [r for r in current_rules if r.enabled]
                    for t in threats:
                        # ALWAYS include Watchlist matches in final_threats
                        if t['object'].startswith('Watchlist:'):
                            final_threats.append(t)
                            continue

                        for rule in enabled_rules:
                            # Simple target matching: if target is in rule target (person, weapon, etc)
                            if rule.target.lower() in t['object'].lower() or t['object'].lower() in rule.target.lower():
                                t['level'] = rule.alertSeverity.capitalize()
                                final_threats.append(t)
                                break
                else:
                    final_threats = threats

                for t in final_threats:
                    alert_key = t['object']
                    if alert_key not in last_alert_time or (current_time - last_alert_time[alert_key] > 5):
                        last_alert_time[alert_key] = current_time
                        
                        event = EventDB(
                            camera_source=str(self.source),
                            detected_object=t['object'],
                            threat_level=t['level'],
                            confidence=t['confidence'],
                            location_x=t['center'][0],
                            location_y=t['center'][1]
                        )
                        db.add(event)
                        db.commit()
                        db.refresh(event)
                        
                        new_alerts.append({
                            'id': event.id,
                            'timestamp': event.timestamp.isoformat(),
                            'object': event.detected_object,
                            'level': event.threat_level
                        })

                with self.lock:
                    self.current_frame = annotated_frame.copy()
                    if new_alerts:
                        self.latest_alerts = (new_alerts + self.latest_alerts)[:20]
                        
                # Track FPS
                now = time.time()
                self._frame_times.append(now)
                self._frame_times = [t for t in self._frame_times if now - t < 1.0]
                self.current_fps = len(self._frame_times)

                # Generate heatmap frame
                try:
                    heatmap_img = self._render_heatmap_overlay(annotated_frame)
                    with self.lock:
                        self.heatmap_frame = heatmap_img
                except Exception:
                    pass

                # Yield processing power
                time.sleep(0.03)
        finally:
            cap.release()
            db.close()

    def get_frame_bytes(self):
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame
        
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def get_heatmap_frame_bytes(self):
        with self.lock:
            if self.heatmap_frame is None:
                return None
            frame = self.heatmap_frame
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def _render_heatmap_overlay(self, base_frame):
        height, width = base_frame.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        for x, y in self.heatmap_data:
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap)) * 255
        heatmap = heatmap.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base_frame, 0.8, heatmap_colored, 0.2, 0)
        return overlay

    def generate_heatmap(self):
        with self.lock:
            if self.current_frame is None:
                return b''
            height, width = self.current_frame.shape[:2]
            
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for x, y in self.heatmap_data:
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1
                
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap)) * 255
            
        heatmap = heatmap.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        ret, buffer = cv2.imencode('.jpg', heatmap_colored)
        return buffer.tobytes()
