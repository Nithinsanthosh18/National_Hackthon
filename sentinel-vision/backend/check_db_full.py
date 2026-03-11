from database import SessionLocal
import models

db = SessionLocal()
try:
    print("--- WATCHLIST ---")
    items = db.query(models.WatchlistDB).all()
    for i in items: print(f"{i.id} | {i.name} | {i.image_path}")
    
    print("\n--- CAMERAS ---")
    cams = db.query(models.CameraDB).all()
    for c in cams: print(f"{c.id} | {c.name} | {c.url}")
    
    print("\n--- RULES ---")
    rules = db.query(models.RuleDB).all()
    for r in rules: print(f"{r.id} | {r.name} | {r.cameras}")

    print("\n--- EVENTS (Top 5) ---")
    events = db.query(models.EventDB).order_by(models.EventDB.timestamp.desc()).limit(5).all()
    for e in events: print(f"{e.id} | {e.detected_object} | {e.timestamp}")

finally:
    db.close()
