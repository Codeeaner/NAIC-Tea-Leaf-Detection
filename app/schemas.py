"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionBoxSchema(BaseModel):
    """Schema for detection bounding box."""
    
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    
    class Config:
        from_attributes = True


class DetectionResultSchema(BaseModel):
    """Schema for detection result."""
    
    id: int
    session_id: int
    image_name: str
    healthy_count: int
    unhealthy_count: int
    total_count: int
    health_percentage: float
    confidence_threshold: float
    processing_time: float
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    annotated_image: Optional[str] = None
    boxes: List[DetectionBoxSchema] = []
    
    class Config:
        from_attributes = True


class DetectionSessionSchema(BaseModel):
    """Schema for detection session."""
    
    id: int
    name: str
    status: str
    total_images: int
    processed_images: int
    created_at: datetime
    updated_at: datetime
    results: List[DetectionResultSchema] = []
    
    class Config:
        from_attributes = True


class DetectionSessionCreateSchema(BaseModel):
    """Schema for creating a detection session."""
    
    name: str = Field(..., min_length=1, max_length=255)


class DetectionSessionSummarySchema(BaseModel):
    """Schema for detection session summary."""
    
    id: int
    name: str
    status: str
    total_images: int
    processed_images: int
    total_healthy: int = 0
    total_unhealthy: int = 0
    average_health_percentage: float = 0.0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Schema for upload response."""
    
    session_id: int
    uploaded_files: List[str]
    message: str


class DetectionStatsSchema(BaseModel):
    """Schema for overall detection statistics."""
    
    total_sessions: int
    total_images_processed: int
    total_healthy_leaves: int
    total_unhealthy_leaves: int
    average_health_percentage: float
    sessions_today: int
    images_processed_today: int


class ProgressUpdateSchema(BaseModel):
    """Schema for progress updates."""
    
    session_id: int
    processed_images: int
    total_images: int
    current_image: str
    status: str
    percentage: float