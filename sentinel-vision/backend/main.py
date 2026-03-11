from fastapi import FastAPI, Depends, Response, HTTPException, Body, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import database
import models
import video_engine
import archive_processor
import uvicorn
import time
import os
import shutil
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel
import json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

def safe_json_encoder(obj):
    """Sanitize objects to ensure they are JSON serializable without Unicode errors."""
    try:
        return jsonable_encoder(obj)
    except Exception:
        # Fallback for binary or weird data
        return str(obj)

class SafeJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=str, # Crucial fallback
        ).encode("utf-8")

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FACES_DIR = os.path.join(BASE_DIR, "User") 

for d in [UPLOAD_DIR, FACES_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Initialize Database Early
models.Base.metadata.create_all(bind=database.engine)

# Global registry of active camera engines
engines = {}

# Archive Analysis state tracker
archive_jobs = {}

class CameraCreate(BaseModel):
    name: str
    camera_id: str
    url: str
    type: Optional[str] = "IP Camera"
    format: Optional[str] = "H.264"

class RuleCreate(BaseModel):
    name: str
    category: str
    target: str
    cameras: str
    confidenceThreshold: float
    alertSeverity: str
    description: str

class RuleUpdate(BaseModel):
    enabled: int

def sync_rules_to_engines(db: Session):
    try:
        rules = db.query(models.RuleDB).all()
        for cam_id, engine in engines.items():
            # Filter rules applicable to this camera or "All"
            cam_rules = [r for r in rules if r.cameras == "All" or cam_id in r.cameras]
            engine.set_rules(cam_rules)
    except Exception as e:
        print(f"Warning: Could not sync rules (table might be initializing): {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    
    # Initialize from DB
    try:
        db = next(database.get_db())
        db_cameras = db.query(models.CameraDB).filter(models.CameraDB.active == 1).all()
        rules = db.query(models.RuleDB).all()
        
        for cam in db_cameras:
            try:
                engine = video_engine.VideoEngine(source=cam.url)
                # Apply rules
                cam_rules = [r for r in rules if r.cameras == "All" or cam.id in r.cameras]
                engine.set_rules(cam_rules)
                
                engine.start()
                engines[cam.id] = engine
                print(f"Started engine for {cam.name} ({cam.id})")
            except Exception as e:
                print(f"Failed to start engine for {cam.id}: {e}")
    except Exception as e:
        print(f"CRITICAL: Database initialization failed in lifespan: {e}")
    
    yield
    
    # Clean up
    for engine in engines.values():
        if engine:
            engine.stop()
    time.sleep(0.5)

app = FastAPI(title="Sentinel Vision API", lifespan=lifespan, default_response_class=SafeJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gen_frames(camera_id: str, heatmap: bool = False):
    print(f"DEBUG: Starting gen_frames for {camera_id} (heatmap={heatmap})")
    engine = engines.get(camera_id)
    if not engine:
        print(f"DEBUG: No engine found for {camera_id}")
        return

    while True:
        if heatmap:
            frame_bytes = engine.get_heatmap_frame_bytes()
        else:
            frame_bytes = engine.get_frame_bytes()
            
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.get("/video-feed/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in engines:
        # Try to find by name if ID fails
        found_id = None
        for eid in engines:
            if camera_id.lower() in eid.lower():
                found_id = eid
                break
        if not found_id:
            return Response(status_code=404, content="Camera not found")
        camera_id = found_id
        
    print(f"DEBUG: Serving video-feed for {camera_id}")
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/heatmap-feed/{camera_id}")
def heatmap_feed(camera_id: str):
    if camera_id not in engines:
        return Response(status_code=404, content="Camera not found")
    return StreamingResponse(gen_frames(camera_id, heatmap=True), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
def get_live_alerts():
    all_alerts = []
    for eng in engines.values():
        all_alerts.extend(eng.latest_alerts)
    all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    return {"alerts": all_alerts[:20]}

@app.get("/events")
def get_events(limit: int = 50, db: Session = Depends(database.get_db)):
    events = db.query(models.EventDB).order_by(models.EventDB.timestamp.desc()).limit(limit).all()
    return {"events": events}

@app.get("/events/count")
def get_event_count(db: Session = Depends(database.get_db)):
    count = db.query(models.EventDB).count()
    return {"count": count}

@app.get("/stats")
def get_stats(db: Session = Depends(database.get_db)):
    try:
        stats = {
            "PERSON": db.query(models.EventDB).filter(models.EventDB.detected_object == "person").count(),
            "WEAPON": db.query(models.EventDB).filter(models.EventDB.detected_object.in_(["gun", "knife"])).count(),
            "WATCHLIST": db.query(models.EventDB).filter(models.EventDB.detected_object.like("Watchlist:%")).count(),
            "OTHER": db.query(models.EventDB).filter(~models.EventDB.detected_object.in_(["person", "gun", "knife"])).filter(~models.EventDB.detected_object.like("Watchlist:%")).count()
        }
        return stats
    except Exception as e:
        print(f"Error in get_stats: {e}")
        return {"PERSON": 0, "WEAPON": 0, "WATCHLIST": 0, "OTHER": 0}

@app.get("/stats/hourly")
def get_hourly_stats():
    # Placeholder for hourly data
    return {"labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"], "data": [5, 2, 8, 15, 12, 7]}

@app.get("/stats/weekly")
def get_weekly_stats():
    # Placeholder for weekly data
    return {"labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "data": [45, 52, 38, 65, 48, 30, 25]}

# Cameras Management
@app.get("/cameras")
def list_cameras(db: Session = Depends(database.get_db)):
    cams = db.query(models.CameraDB).all()
    cam_list = []
    for c in cams:
        eng = engines.get(c.id)
        cam_list.append({
            "id": c.id,
            "name": c.name,
            "camId": c.id,
            "type": c.type,
            "url": c.url,
            "active": c.active == 1,
            "format": c.format,
            "fps": eng.current_fps if eng else 0
        })
    # Also provide a dictionary keyed by ID for the frontend specifically for campaign metrics if needed
    cam_dict = {c["id"]: c for c in cam_list}
    return {"cameras": cam_list, "cameras_dict": cam_dict, "active_cameras": len([e for e in engines.values() if e.running])}

@app.post("/cameras")
def add_camera(cam: CameraCreate, db: Session = Depends(database.get_db)):
    url = cam.url.strip()
    
    # Better heuristic for IP addresses and common camera apps
    import re
    # Match raw IP or IP:PORT or http(s)://IP:PORT
    ip_pattern = r"^(?:https?://)?(\d{1,3}(\.\d{1,3}){3})(?::(\d+))?/?$"
    match = re.match(ip_pattern, url)
    
    suggestion = None
    if match:
        ip = match.group(1)
        port = match.group(2)
        print(f"Detected IP format: {url}")
        if port == "8080":
            suggestion = f"http://{ip}:8080/video"
            print(f"TIP: This looks like 'IP Webcam' app. Try using {suggestion}")
        elif port == "4747":
            suggestion = f"http://{ip}:4747/video"
            print(f"TIP: This looks like 'DroidCam'. Try using {suggestion}")
        else:
            print(f"TIP: For IP cameras, you usually need a stream path (e.g., rtsp://{ip}/live or http://{ip}/video)")
    
    db_cam = models.CameraDB(id=cam.camera_id, name=cam.name, url=url, type=cam.type, format=cam.format, active=1)
    db.add(db_cam)
    db.commit()
    
    try:
        engine = video_engine.VideoEngine(source=url)
        engine.start()
        engines[cam.camera_id] = engine
        
        msg = "Camera added successfully"
        if suggestion and not engine.running:
             msg += f". NOTE: Connection failed. If this is a mobile app, try using: {suggestion}"
             
        return {"status": "ok", "message": msg}
    except Exception as e:
        print(f"Failed to start camera {url}: {e}")
        error_msg = f"Could not open camera: {str(e)}"
        if suggestion:
            error_msg += f". Try using the stream path instead: {suggestion}"
        return {"status": "error", "message": error_msg}

@app.post("/purge-db")
def purge_database(db: Session = Depends(database.get_db)):
    """Utility to clear all events and watchlist items if data is corrupt."""
    db.query(models.EventDB).delete()
    db.query(models.WatchlistDB).delete()
    db.commit()
    # Also clear face folders
    if os.path.exists(FACES_DIR):
        shutil.rmtree(FACES_DIR)
        os.makedirs(FACES_DIR)
    return {"status": "Database and face data purged"}

@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str, db: Session = Depends(database.get_db)):
    db_cam = db.query(models.CameraDB).filter(models.CameraDB.id == cam_id).first()
    if db_cam:
        db.delete(db_cam)
        db.commit()
    
    if cam_id in engines:
        engines[cam_id].stop()
        del engines[cam_id]
    return {"status": "ok"}

# Watchlist Management
@app.get("/watchlist")
def get_watchlist(db: Session = Depends(database.get_db)):
    items = db.query(models.WatchlistDB).all()
    suspects = []
    for item in items:
        suspects.append({
            "id": item.id,
            "name": item.name,
            "alias": item.alias or "None",
            "threat": item.threat or "Medium",
            "status": item.status or "Active",
            "notes": item.notes or "",
            "added_date": item.added_date.isoformat() if item.added_date else None,
            "image_url": f"/uploads/{os.path.basename(item.image_path)}" if item.image_path else None
        })
    return {"suspects": suspects}

@app.post("/watchlist")
async def add_watchlist(
    name: str = Form(...), 
    alias: Optional[str] = Form(None),
    threat: Optional[str] = Form("Medium"),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...), 
    db: Session = Depends(database.get_db)
):
    file_id = str(int(time.time()))
    file_ext = os.path.splitext(file.filename)[1]
    print(f"Watchlist Upload received: {name}, file: {file.filename}")
    
    # Save to FACES_DIR structure: FACES_DIR/[name]/[id].[ext]
    user_folder = os.path.join(FACES_DIR, name.replace(" ", "_"))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        
    file_path = os.path.join(user_folder, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Also copy to uploads for UI display
    ui_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    shutil.copy(file_path, ui_path)
        
    db_item = models.WatchlistDB(
        id=file_id, 
        name=name, 
        alias=alias,
        threat=threat,
        notes=notes,
        status="Active",
        image_path=ui_path
    )
    db.add(db_item)
    db.commit()
    
    # Trigger re-training for all engines
    print(f"Triggering re-training for {len(engines)} engines...")
    for eng in engines.values():
        eng.update_face_recognizer()
        
    print("Watchlist item added and engines notified.")
    return {"status": "ok"}

@app.delete("/watchlist/{item_id}")
def delete_watchlist(item_id: str, db: Session = Depends(database.get_db)):
    db_item = db.query(models.WatchlistDB).filter(models.WatchlistDB.id == item_id).first()
    if db_item:
        name = db_item.name
        # Delete from FACES_DIR structure: FACES_DIR/[name]
        user_folder = os.path.join(FACES_DIR, name.replace(" ", "_"))
        if os.path.exists(user_folder):
            print(f"Deleting face training folder: {user_folder}")
            shutil.rmtree(user_folder)
            
        # Delete from uploads
        if db_item.image_path and os.path.exists(db_item.image_path):
            os.remove(db_item.image_path)
            
        db.delete(db_item)
        db.commit()
        
        # Trigger re-training for all engines
        print("Triggering re-training after suspect deletion...")
        for eng in engines.values():
            eng.update_face_recognizer()
            
    return {"status": "ok"}

from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Rules Management
@app.get("/rules")
def get_rules(db: Session = Depends(database.get_db)):
    rules = db.query(models.RuleDB).all()
    return {"rules": rules}

@app.post("/rules")
def create_rule(rule: RuleCreate, db: Session = Depends(database.get_db)):
    rule_id = str(int(time.time()))
    db_rule = models.RuleDB(
        id=rule_id,
        name=rule.name,
        category=rule.category,
        target=rule.target,
        cameras=rule.cameras,
        confidenceThreshold=rule.confidenceThreshold,
        alertSeverity=rule.alertSeverity,
        enabled=1,
        description=rule.description
    )
    db.add(db_rule)
    db.commit()
    
    # Sync to engines
    sync_rules_to_engines(db)
    return {"status": "ok"}

@app.patch("/rules/{rule_id}")
def update_rule(rule_id: str, update: RuleUpdate, db: Session = Depends(database.get_db)):
    db_rule = db.query(models.RuleDB).filter(models.RuleDB.id == rule_id).first()
    if not db_rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    db_rule.enabled = update.enabled
    db.commit()
    
    # Sync to engines
    sync_rules_to_engines(db)
    return {"status": "ok"}

@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: str, db: Session = Depends(database.get_db)):
    db_rule = db.query(models.RuleDB).filter(models.RuleDB.id == rule_id).first()
    if db_rule:
        db.delete(db_rule)
        db.commit()
    
    # Sync to engines
    sync_rules_to_engines(db)
    return {"status": "ok"}

# Archive Tracking
@app.post("/archive/analyze")
async def analyze_archive(file: UploadFile = File(...)):
    job_id = "job_" + str(int(time.time()))
    file_ext = os.path.splitext(file.filename)[1]
    input_filename = f"{job_id}_input{file_ext}"
    output_filename = f"{job_id}_output.mp4"
    
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    archive_jobs[job_id] = {"status": "Processing", "progress": 0}
    
    # Process in a background thread to avoid blocking FastAPI
    import threading
    def process_task():
        try:
            archive_processor.processor.process_video(input_path, output_path, job_id, archive_jobs)
        except Exception as e:
            print(f"Error processing archive {job_id}: {e}")
            archive_jobs[job_id]["status"] = "Failed"
            
    threading.Thread(target=process_task).start()
    return {"job_id": job_id}

@app.get("/archive/status/{job_id}")
async def archive_status(job_id: str):
    job = archive_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job["status"] == "Completed":
        return {
            "status": "Completed", 
            "findings": job.get("findings", []), 
            "video_url": job.get("video_url")
        }
    return job

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
