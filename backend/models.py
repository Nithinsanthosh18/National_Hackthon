from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class EventDB(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_source = Column(String, index=True)
    detected_object = Column(String)
    threat_level = Column(String)  # e.g., 'low', 'medium', 'high', 'critical'
    confidence = Column(Float)
    location_x = Column(Integer, nullable=True) # Center x
    location_y = Column(Integer, nullable=True) # Center y
    image_path = Column(String, nullable=True) # Optional path to saved frame crop

class CameraDB(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    url = Column(String)
    type = Column(String)
    format = Column(String)
    active = Column(Integer, default=1)

class WatchlistDB(Base):
    __tablename__ = "watchlist"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    image_path = Column(String)
    # Storing PyTorch tensor/numpy array as binary
    face_encoding = Column(String) # We will actually store base64 encoded strings or serialized bytes here

class RuleDB(Base):
    __tablename__ = "rules"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    category = Column(String) # Object, Behavior, Zone, Facial
    target = Column(String) # e.g. "person", "cell phone"
    cameras = Column(String) # Comma-separated list: "cam1,cam2"
    confidence_threshold = Column(Float)
    alert_severity = Column(String) # HIGH, MEDIUM, LOW
    enabled = Column(Integer, default=1)
    description = Column(String)
