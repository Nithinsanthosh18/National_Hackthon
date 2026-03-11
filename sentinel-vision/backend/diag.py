import os
from sqlalchemy import select
from database import SessionLocal
from models import WatchlistDB

db = SessionLocal()
try:
    items = db.query(WatchlistDB).all()
    print(f"Total Watchlist Items: {len(items)}")
    for item in items:
        print(f"ID: {item.id}, Name: {item.name}, Image Path: {item.image_path}")
finally:
    db.close()

# Also check directory absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "User")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
print(f"BASE_DIR: {BASE_DIR}")
print(f"FACES_DIR: {FACES_DIR}")
print(f"UPLOAD_DIR: {UPLOAD_DIR}")
print(f"Does FACES_DIR exist? {os.path.exists(FACES_DIR)}")
print(f"Does UPLOAD_DIR exist? {os.path.exists(UPLOAD_DIR)}")
