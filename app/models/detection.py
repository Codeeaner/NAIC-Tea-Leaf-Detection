"""Database models for tea leaf detection results."""

import base64
from datetime import datetime
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


class DetectionSession(Base):
    """Model for detection sessions."""
    
    __tablename__ = "detection_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    results = relationship("DetectionResult", back_populates="session", cascade="all, delete-orphan")
    batch_analytics = relationship("BatchAnalysisResult", back_populates="session", cascade="all, delete-orphan")


class DetectionResult(Base):
    """Model for individual detection results."""
    
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(255), nullable=False)
    healthy_count = Column(Integer, default=0)
    unhealthy_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    health_percentage = Column(Float, default=0.0)
    confidence_threshold = Column(Float, default=0.25)
    processing_time = Column(Float, default=0.0)
    annotated_image_path = Column(String(500))
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("DetectionSession", back_populates="results")
    boxes = relationship("DetectionBox", back_populates="result", cascade="all, delete-orphan")

    @property
    def annotated_image(self):
        """Return the annotated image as a base64 string when the file exists."""
        if not self.annotated_image_path:
            return None

        image_path = Path(self.annotated_image_path)
        if not image_path.exists():
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class DetectionBox(Base):
    """Model for individual detection bounding boxes."""
    
    __tablename__ = "detection_boxes"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("detection_results.id"), nullable=False)
    class_id = Column(Integer, nullable=False)  # 0=unhealthy, 1=healthy
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    
    # Relationships
    result = relationship("DetectionResult", back_populates="boxes")
