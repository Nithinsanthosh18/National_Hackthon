import cv2
import threading
import time
from detector import ThreatDetector
from database import SessionLocal
from models import EventDB, WatchlistDB, RuleDB
from datetime import datetime
import numpy as np
import asyncio
import base64
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize facenet models natively so we can call them in the thread
mtcnn = MTCNN(keep_all=True) # Return all faces in frame
resnet = InceptionResnetV1(pretrained='vggface2').eval()

class VideoEngine:
    def __init__(self, source=0, camera_id=None):
        self.camera_id = camera_id
        # Auto-format common IP Webcam URLs if the user just provides the root
        if isinstance(source, str) and source.startswith("http") and source.count("/") == 2:
            self.source = f"{source}/video"
        elif isinstance(source, str) and source.startswith("http") and source.endswith(":8080"):
            self.source = f"{source}/video"
        elif isinstance(source, str) and source.startswith("http") and source.endswith(":8080/"):
            self.source = f"{source}video"
        else:
            self.source = source
            
        print(f"Initializing VideoEngine with source: {self.source}")
        print("Initializing YOLO ThreatDetector. This may download weights...")
        self.detector = ThreatDetector()
        
        self.current_frame = None
        self.current_heatmap_frame = None
        self.latest_alerts = []
        self.heatmap_data = [] 
        self.current_fps = 0.0
        self.lock = threading.Lock()
        self.known_suspects = []
        self._suspects_lock = threading.Lock()

    def reload_suspects(self):
        """Re-read watchlist from DB and rebuild in-memory suspect list.
        Can be called at any time (thread-safe) to pick up new uploads."""
        db = SessionLocal()
        try:
            suspect_records = db.query(WatchlistDB).all()
            new_suspects = []
            for s in suspect_records:
                if s.face_encoding:
                    b_data = base64.b64decode(s.face_encoding)
                    np_array = np.frombuffer(b_data, dtype=np.float32)
                    tensor = torch.from_numpy(np_array).unsqueeze(0)
                    new_suspects.append({"id": s.id, "name": s.name, "encoding": tensor})
            with self._suspects_lock:
                self.known_suspects = new_suspects
            print(f"Reloaded {len(new_suspects)} watchlisted suspects into VideoEngine memory.")
        except Exception as e:
            print(f"Error reloading suspects: {e}")
        finally:
            db.close()

    def reload_rules(self):
        """Fetch rules from DB and apply to the detector."""
        db = SessionLocal()
        try:
            # Get all enabled rules
            all_rules = db.query(WatchlistDB if False else RuleDB).filter(RuleDB.enabled == 1).all()
            
            # Filter for this specific camera if camera_id is set
            camera_rules = []
            for r in all_rules:
                if not r.cameras or self.camera_id in [c.strip() for c in r.cameras.split(',')]:
                    camera_rules.append(r)
            
            self.detector.set_rules(camera_rules)
            print(f"Engine {self.camera_id}: Loaded {len(camera_rules)} detection rules.")
        except Exception as e:
            print(f"Error reloading rules: {e}")
        finally:
            db.close()

    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def stop(self):
        self.running = False

    def _update(self):
        print("Starting video engine background thread...")
        
        # Load detector FIRST before locking the camera so downloads don't hold the lock
        db = SessionLocal()
        last_alert_time = {}
        last_face_scan_time = 0
        
        # Pre-load known suspects using the shared reload method
        self.reload_suspects()
        self.reload_rules()
        
        # Initialize video
        if isinstance(self.source, int) or str(self.source).isdigit():
            print(f"Opening camera {self.source} with DSHOW backend...")
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            print(f"Opening video source {self.source}...")
            cap = cv2.VideoCapture(self.source)
            
        print(f"Camera opened: {cap.isOpened()}")    
        fps_start_time = time.time()
        frame_count = 0

        try:
            while self.running:
                if not cap.isOpened():
                    print("Camera not open, retrying...")
                    time.sleep(1)
                    if isinstance(self.source, int) or str(self.source).isdigit():
                        cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(self.source)
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera")
                    
                    # For network streams (like IP Webcam), cv2 can sometimes drop frames
                    # or need a few seconds to buffer. We should wait and retry instead
                    # of instantly failing.
                    if isinstance(self.source, str) and (self.source.startswith('http') or self.source.startswith('rtsp')):
                        time.sleep(2)
                        # Re-initialize the capture object to force connection refresh
                        cap = cv2.VideoCapture(self.source)
                    elif isinstance(self.source, str) and not self.source.startswith('rtsp') and not self.source.startswith('http'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(1)
                    continue
                
                # Process threats
                try:
                    annotated_frame, threats, persons = self.detector.process_frame(frame)
                except Exception as e:
                    print(f"Error during YOLO processing: {e}")
                    continue
                
                self.heatmap_data.extend(persons)
                self.heatmap_data = self.heatmap_data[-2000:] # Cap heatmap data size

                new_alerts = []
                current_time = time.time()
                
                for t in threats:
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

                # Check for Watchlist Suspects (Running every ~0.5s to save CPU)
                # Grab a thread-safe snapshot of the current suspects list
                with self._suspects_lock:
                    current_suspects = list(self.known_suspects)

                if current_suspects and current_time - last_face_scan_time > 0.5:
                    last_face_scan_time = current_time
                    try:
                        # Convert CV2 BGR to RGB PIL Image
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        
                        # Detect faces
                        boxes, _ = mtcnn.detect(pil_img)
                        if boxes is not None:
                            # Extract face crops into tensors
                            faces = mtcnn(pil_img)
                            if faces is not None:
                                # If there's only one face, MTCNN returns it natively, otherwise it returns a stack. Fix dimension
                                if len(faces.shape) == 3:
                                    faces = faces.unsqueeze(0)
                                    
                                # Get embeddings for all detected faces
                                embeddings = resnet(faces).detach()
                                
                                # Compare each live face to our watchlist
                                for i, face_emb in enumerate(embeddings):
                                    for suspect in current_suspects:
                                        # Calculate Euclidean distance
                                        dist = (face_emb - suspect["encoding"].squeeze(0)).norm().item()
                                        
                                        # Threshold for match (typically < 0.8 is a match for vggface2 resnet)
                                        if dist < 0.8:
                                            # We have a match! Draw box
                                            box = boxes[i]
                                            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                                            cv2.putText(annotated_frame, f"SUSPECT: {suspect['name']}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                            
                                            alert_key = f"suspect_{suspect['id']}"
                                            if alert_key not in last_alert_time or (current_time - last_alert_time[alert_key] > 5):
                                                last_alert_time[alert_key] = current_time
                                                event = EventDB(
                                                    camera_source=str(self.source),
                                                    detected_object=f"Suspect: {suspect['name']}",
                                                    threat_level="critical",
                                                    confidence=round(1.0 - (dist/2.0), 2),
                                                    location_x=int((box[0]+box[2])/2),
                                                    location_y=int((box[1]+box[3])/2)
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
                    except Exception as e:
                        print(f"Face processing error: {e}")

                # Use a solid black background instead of the video frame
                heatmap_annotated_frame = np.zeros_like(annotated_frame)

                # Generate heatmap overlay
                if len(self.heatmap_data) > 0:
                    height, width = heatmap_annotated_frame.shape[:2]
                    
                    # Create blank matrix
                    heatmap_matrix = np.zeros((height, width), dtype=np.float32)
                    
                    # Accumulate valid points
                    for x, y in self.heatmap_data:
                        if 0 <= x < width and 0 <= y < height:
                            heatmap_matrix[y, x] += 1
                            
                    # Smooth and normalize
                    heatmap_matrix = cv2.GaussianBlur(heatmap_matrix, (51, 51), 0)
                    max_val = np.max(heatmap_matrix)
                    if max_val > 0:
                        heatmap_matrix = (heatmap_matrix / max_val) * 255
                        
                    # Apply colormap
                    heatmap_matrix = heatmap_matrix.astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_matrix, cv2.COLORMAP_JET)
                    
                    # Create alpha mask (only overlay where there is heat)
                    # We create a mask based on the intensity of the heatmap
                    # Areas with 0 intensity (cold) will not be overlaid
                    alpha_mask = (heatmap_matrix > 5).astype(np.float32)
                    alpha_mask = np.dstack([alpha_mask]*3) # Make it 3-channel
                    
                    # Blend the image with the black background
                    alpha = 0.8 # Make heatmap highly visible since there's no background
                    # Only blend the regions where mask is 1
                    blended = cv2.addWeighted(heatmap_annotated_frame, 1 - alpha, heatmap_colored, alpha, 0)
                    heatmap_annotated_frame = np.where(alpha_mask > 0, blended, heatmap_annotated_frame)

                with self.lock:
                    self.current_frame = annotated_frame.copy()
                    self.current_heatmap_frame = heatmap_annotated_frame.copy()
                    if new_alerts:
                        self.latest_alerts = (new_alerts + self.latest_alerts)[:20]

                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    self.current_fps = 10 / (now - fps_start_time)
                    fps_start_time = now
                        
                # Yield processing power
                time.sleep(0.01)
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
            if self.current_heatmap_frame is None:
                return None
            frame = self.current_heatmap_frame
        
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

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
