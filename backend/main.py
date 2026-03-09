from fastapi import FastAPI, Depends, Response, File, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import asyncio
import database
import models
import video_engine
import uvicorn
import time
import os
import shutil
import uuid
import base64
from contextlib import asynccontextmanager
from archive_processor import process_video_file
from pydantic import BaseModel
import PIL.Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize Face Recognition Models globally
mtcnn = MTCNN(keep_all=False, select_largest=True) # Only grab the biggest face in the uploaded ID photo
resnet = InceptionResnetV1(pretrained='vggface2').eval()
from contextlib import asynccontextmanager
from archive_processor import process_video_file
from pydantic import BaseModel

class CameraCreate(BaseModel):
    name: str
    camera_id: str
    url: str
    type: str = "IP Camera"
    format: str = "H.264"

class RuleCreate(BaseModel):
    name: str
    category: str
    target: str
    cameras: str = "" # Comma separated
    confidenceThreshold: float = 0.7
    alertSeverity: str = "HIGH"
    description: str = ""

class RuleUpdate(BaseModel):
    name: str = None
    category: str = None
    target: str = None
    cameras: str = None
    confidenceThreshold: float = None
    alertSeverity: str = None
    enabled: int = None
    description: str = None

# We will no longer rely on a hardcoded dict.
# CAMERA_SOURCES is kept as an in-memory cache of urls.
CAMERA_SOURCES = {}


# Global registry of active camera engines
engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    
    # Initialize database if not exists
    models.Base.metadata.create_all(bind=database.engine)
    
    # Load cameras from DB
    db = database.SessionLocal()
    saved_cameras = db.query(models.CameraDB).filter(models.CameraDB.active == 1).all()
    
    # If no cameras exist, add a default one (like the old hardcoded cam1)
    if not saved_cameras:
        default_cam = models.CameraDB(
            id="cam1", name="Main Entrance", url="0", type="Webcam", format="H.264", active=1
        )
        db.add(default_cam)
        db.commit()
        saved_cameras = [default_cam]

    for cam in saved_cameras:
        source = cam.url
        if source.isdigit():
            source = int(source)
            
        CAMERA_SOURCES[cam.id] = source
        engine = video_engine.VideoEngine(source=source, camera_id=cam.id)
        engine.start()
        engine.reload_suspects()
        engine.reload_rules()
        engines[cam.id] = engine
        
    db.close()
    
    yield
    
    # Clean up all cameras gracefully
    for engine in engines.values():
        if engine:
            engine.stop()
    time.sleep(0.5)

app = FastAPI(title="Sentinel Vision API", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("archive_outputs", exist_ok=True)
app.mount("/archive_outputs", StaticFiles(directory="archive_outputs"), name="archive_outputs")

# Serve UI Static Files
UI_DIST = os.path.abspath("../frontend/dist")
if os.path.exists(UI_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(UI_DIST, "assets")), name="assets")

os.makedirs("watchlist_images", exist_ok=True)
app.mount("/watchlist_images", StaticFiles(directory="watchlist_images"), name="watchlist_images")

archive_jobs = {}

archive_jobs = {}

def gen_frames(camera_id: str):
    engine = engines.get(camera_id)
    if not engine:
        return

    while True:
        frame_bytes = engine.get_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

def gen_heatmap_frames(camera_id: str):
    engine = engines.get(camera_id)
    if not engine:
        return

    while True:
        frame_bytes = engine.get_heatmap_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.get("/video-feed/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in engines:
        return Response(status_code=404, content="Camera not found")
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/heatmap-feed/{camera_id}")
def heatmap_feed(camera_id: str):
    if camera_id not in engines:
        return Response(status_code=404, content="Camera not found")
    return StreamingResponse(gen_heatmap_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
def get_live_alerts():
    # Aggregate alerts from all cameras, sort by time, and get the latest 20
    all_alerts = []
    for eng in engines.values():
        all_alerts.extend(eng.latest_alerts)
    
    # Sort descending by timestamp
    all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    return {"alerts": all_alerts[:20]}

@app.get("/events")
def get_events(limit: int = 50, db: Session = Depends(database.get_db)):
    events = db.query(models.EventDB).order_by(models.EventDB.timestamp.desc()).limit(limit).all()
    return {"events": events}

@app.get("/events/count")
def get_events_count(db: Session = Depends(database.get_db)):
    count = db.query(models.EventDB).count()
    return {"count": count}

@app.get("/heatmap/{camera_id}")
def get_heatmap(camera_id: str):
    engine = engines.get(camera_id)
    if not engine:
        return Response(status_code=404, content="Camera not found")
        
    heatmap_bytes = engine.generate_heatmap()
    return Response(content=heatmap_bytes, media_type="image/jpeg")

@app.get("/cameras")
def get_cameras(db: Session = Depends(database.get_db)):
    # Get metadata from DB, overlay dynamic FPS
    db_cams = db.query(models.CameraDB).all()
    
    cameras_list = []
    for cam in db_cams:
        engine = engines.get(cam.id)
        fps = round(engine.current_fps, 1) if engine else 0.0
        cameras_list.append({
            "id": cam.id,
            "name": cam.name,
            "camId": cam.id,
            "type": cam.type,
            "url": cam.url,
            "active": bool(cam.active),
            "format": cam.format,
            "fps": fps
        })
        
    return {"active_cameras": len(engines), "cameras": cameras_list}

@app.post("/cameras")
def add_camera(cam: CameraCreate, db: Session = Depends(database.get_db)):
    source = cam.url
    if source.isdigit():
        source = int(source)
        
    try:
        # Save to DB first
        db_cam = models.CameraDB(
            id=cam.camera_id,
            name=cam.name,
            url=cam.url,
            type=cam.type,
            format=cam.format,
            active=1
        )
        db.add(db_cam)
        db.commit()

        # Start dynamic engine
        engine = video_engine.VideoEngine(source=source, camera_id=cam.camera_id)
        engine.start()
        engine.reload_suspects()
        engine.reload_rules()
        engines[cam.camera_id] = engine
        CAMERA_SOURCES[cam.camera_id] = source
        return {"status": "success", "camera_id": cam.camera_id}
    except Exception as e:
        return Response(status_code=500, content=f"Failed to start camera: {e}")

@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str, db: Session = Depends(database.get_db)):
    # 1. Remove from database
    db_cam = db.query(models.CameraDB).filter(models.CameraDB.id == camera_id).first()
    if db_cam:
        db.delete(db_cam)
        db.commit()
        
    # 2. Stop Python thread
    if camera_id in engines:
        engines[camera_id].stop()
        del engines[camera_id]
        
    if camera_id in CAMERA_SOURCES:
        del CAMERA_SOURCES[camera_id]
        
    return {"status": "success", "message": f"Camera {camera_id} removed"}

@app.get("/watchlist")
def get_watchlist(db: Session = Depends(database.get_db)):
    suspects = db.query(models.WatchlistDB).all()
    return {"suspects": [{"id": s.id, "name": s.name, "image_url": f"http://127.0.0.1:8001/{s.image_path}"} for s in suspects]}

@app.post("/watchlist")
async def add_suspect(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    suspect_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    filename = f"{suspect_id}.{ext}"
    filepath = os.path.join("watchlist_images", filename)
    
    # Save image
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load image for processing
        img = PIL.Image.open(filepath)
        # Extract face
        face_tensor = mtcnn(img)
        
        if face_tensor is None:
            os.remove(filepath)
            return Response(status_code=400, content="No face detected in the provided image. Please upload a clear photo.")
            
        # Extract encoding
        # Add batch dimension to the tensor
        embeddings = resnet(face_tensor.unsqueeze(0)).detach().numpy()
        
        # Serialize the numpy array back to base64 so we can save it in SQLite easily
        embedding_b64 = base64.b64encode(embeddings.tobytes()).decode('utf-8')
        
        # Save to DB
        suspect = models.WatchlistDB(
            id=suspect_id,
            name=name,
            image_path=filepath,
            face_encoding=embedding_b64
        )
        db.add(suspect)
        db.commit()
        
        # Notify all running engines to pick up the new suspect immediately
        for engine in engines.values():
            engine.reload_suspects()
        
        return {"status": "success", "id": suspect_id}
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return Response(status_code=500, content=f"Error processing face: {e}")

@app.delete("/watchlist/{suspect_id}")
def delete_suspect(suspect_id: str, db: Session = Depends(database.get_db)):
    suspect = db.query(models.WatchlistDB).filter(models.WatchlistDB.id == suspect_id).first()
    if suspect:
        if os.path.exists(suspect.image_path):
            os.remove(suspect.image_path)
        db.delete(suspect)
        db.commit()
        
        # Notify all running engines to remove this suspect from recognition
        for engine in engines.values():
            engine.reload_suspects()
        
        return {"status": "success"}
    return Response(status_code=404, content="Suspect not found")

from sqlalchemy import func

@app.get("/stats")
def get_stats(db: Session = Depends(database.get_db)):
    # Group detections by object class
    stats = db.query(
        models.EventDB.detected_object, 
        func.count(models.EventDB.id)
    ).group_by(models.EventDB.detected_object).all()
    
    # Convert to dict
    result = {item[0]: item[1] for item in stats}
    return result

@app.get("/stats/hourly")
def get_stats_hourly(db: Session = Depends(database.get_db)):
    # Get detections for the last 24 hours
    now = datetime.now()
    hourly_stats = []
    
    for i in range(24):
        hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
        hour_end = hour_start + timedelta(hours=1)
        
        detections = db.query(models.EventDB).filter(
            models.EventDB.timestamp >= hour_start,
            models.EventDB.timestamp < hour_end
        ).count()
        
        alerts = db.query(models.EventDB).filter(
            models.EventDB.timestamp >= hour_start,
            models.EventDB.timestamp < hour_end,
            models.EventDB.level == 'high'
        ).count()
        
        hourly_stats.append({
            "hour": hour_start.strftime("%H:00"),
            "detections": detections,
            "alerts": alerts
        })
        
    return hourly_stats[::-1] # Return in chronological order

@app.get("/stats/weekly")
def get_stats_weekly(db: Session = Depends(database.get_db)):
    # Get detections for the last 7 days
    now = datetime.now()
    weekly_stats = []
    
    for i in range(7):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        persons = db.query(models.EventDB).filter(
            models.EventDB.timestamp >= day_start,
            models.EventDB.timestamp < day_end,
            models.EventDB.detected_object.ilike('%person%')
        ).count()
        
        vehicles = db.query(models.EventDB).filter(
            models.EventDB.timestamp >= day_start,
            models.EventDB.timestamp < day_end,
            models.EventDB.detected_object.ilike('%vehicle%')
        ).count()
        
        alerts = db.query(models.EventDB).filter(
            models.EventDB.timestamp >= day_start,
            models.EventDB.timestamp < day_end,
            models.EventDB.level == 'high'
        ).count()
        
        weekly_stats.append({
            "day": day_start.strftime("%a"),
            "person": persons,
            "vehicle": vehicles,
            "alert": alerts
        })
        
    return weekly_stats[::-1]

@app.post("/archive/analyze")
async def analyze_archive(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    input_path = os.path.join("archive_outputs", f"{job_id}_input.mp4")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    archive_jobs[job_id] = {"status": "Processing", "findings": [], "video_url": None}
    
    def process_task():
        output_filename = f"{job_id}_output.webm"
        output_path = os.path.join("archive_outputs", output_filename)
        try:
            findings = process_video_file(input_path, output_path)
            archive_jobs[job_id]["status"] = "Completed"
            archive_jobs[job_id]["findings"] = findings
            archive_jobs[job_id]["video_url"] = f"http://127.0.0.1:8001/archive_outputs/{output_filename}"
        except Exception as e:
            print(f"Archive processing failed: {e}")
            archive_jobs[job_id]["status"] = "Failed"
            
    background_tasks.add_task(process_task)
    return {"job_id": job_id, "status": "Processing"}

@app.get("/archive/status/{job_id}")
def get_archive_status(job_id: str):
    if job_id not in archive_jobs:
        return Response(status_code=404, content="Job not found")
    return archive_jobs[job_id]

# --- Detection Rules Endpoints ---

@app.get("/rules")
def get_rules(db: Session = Depends(database.get_db)):
    rules = db.query(models.RuleDB).all()
    return {"rules": rules}

@app.post("/rules")
def create_rule(rule: RuleCreate, db: Session = Depends(database.get_db)):
    rule_id = str(uuid.uuid4())
    db_rule = models.RuleDB(
        id=rule_id,
        name=rule.name,
        category=rule.category,
        target=rule.target,
        cameras=rule.cameras,
        confidence_threshold=rule.confidenceThreshold,
        alert_severity=rule.alertSeverity,
        description=rule.description,
        enabled=1
    )
    db.add(db_rule)
    db.commit()
    
    # Notify all engines
    for engine in engines.values():
        engine.reload_rules()
        
    return {"status": "success", "id": rule_id}

@app.patch("/rules/{rule_id}")
def update_rule(rule_id: str, rule: RuleUpdate, db: Session = Depends(database.get_db)):
    db_rule = db.query(models.RuleDB).filter(models.RuleDB.id == rule_id).first()
    if not db_rule:
        return Response(status_code=404, content="Rule not found")
    
    if rule.name is not None: db_rule.name = rule.name
    if rule.category is not None: db_rule.category = rule.category
    if rule.target is not None: db_rule.target = rule.target
    if rule.cameras is not None: db_rule.cameras = rule.cameras
    if rule.confidenceThreshold is not None: db_rule.confidence_threshold = rule.confidenceThreshold
    if rule.alertSeverity is not None: db_rule.alert_severity = rule.alertSeverity
    if rule.enabled is not None: db_rule.enabled = rule.enabled
    if rule.description is not None: db_rule.description = rule.description
    
    db.commit()
    
    # Notify all engines
    for engine in engines.values():
        engine.reload_rules()
        
    return {"status": "success"}

@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: str, db: Session = Depends(database.get_db)):
    db_rule = db.query(models.RuleDB).filter(models.RuleDB.id == rule_id).first()
    if not db_rule:
        return Response(status_code=404, content="Rule not found")
    
    db.delete(db_rule)
    db.commit()
    
    # Notify all engines
    for engine in engines.values():
        engine.reload_rules()
        
    return {"status": "success"}

# SPA Fallback: Serve index.html for any unknown route (React handles routing)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    ui_index = os.path.join(UI_DIST, "index.html")
    if os.path.exists(ui_index):
        return FileResponse(ui_index)
    return {"error": "UI not found, please run build in UI directory"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
