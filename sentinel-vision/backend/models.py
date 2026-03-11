from sqlalchemy import Column, Integer, String, Float, DateTime, Text
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
    type = Column(String, default="IP Camera")
    format = Column(String, default="H.264")
    active = Column(Integer, default=1)

class WatchlistDB(Base):
    __tablename__ = "watchlist"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    alias = Column(String, nullable=True)
    threat = Column(String, default="Medium")
    status = Column(String, default="Active")
    notes = Column(Text, nullable=True)
    added_date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)
    face_encoding = Column(Text, nullable=True)

class RuleDB(Base):
    __tablename__ = "rules"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    category = Column(String)
    target = Column(String)
    cameras = Column(String)  # Stored as comma-separated string
    confidenceThreshold = Column(Float)
    alertSeverity = Column(String)
    enabled = Column(Integer, default=1)
    description = Column(String)
