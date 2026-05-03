"""Database models for analytics results."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class AnalysisResult(Base):
    """Model for storing AI analysis results."""
    
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(100), unique=True, nullable=False, index=True)
    detection_result_id = Column(Integer, ForeignKey("detection_results.id"), nullable=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=True)
    
    # Analysis metadata
    model_used = Column(String(100), default="qwen3-vl:235b-cloud")
    processing_time = Column(Float, default=0.0)
    status = Column(String(50), default="pending")  # pending, completed, failed, completed_fallback
    error_message = Column(Text)
    
    # Image information
    image_path = Column(String(500))
    annotated_image_path = Column(String(500))
    
    # Detection summary
    healthy_count = Column(Integer, default=0)
    unhealthy_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    health_percentage = Column(Float, default=0.0)
    
    # AI analysis results (stored as JSON)
    ai_analysis = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detection_result = relationship("DetectionResult", backref="analytics")
    session = relationship("DetectionSession", backref="session_analytics")


class BatchAnalysisResult(Base):
    """Model for storing batch-level analysis results."""
    
    __tablename__ = "batch_analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_analysis_id = Column(String(100), unique=True, nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    
    # Batch metadata
    total_images = Column(Integer, default=0)
    analyzed_images = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    status = Column(String(50), default="pending")
    
    # Aggregate statistics
    total_healthy_leaves = Column(Integer, default=0)
    total_unhealthy_leaves = Column(Integer, default=0)
    total_leaves = Column(Integer, default=0)
    overall_health_percentage = Column(Float, default=0.0)
    
    # Batch analysis results (stored as JSON)
    aggregate_recommendations = Column(JSON)
    individual_analysis_ids = Column(JSON)  # List of individual analysis IDs
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("DetectionSession", back_populates="batch_analytics")


class WastePreventionRecommendation(Base):
    """Model for storing specific waste prevention recommendations."""
    
    __tablename__ = "waste_prevention_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    
    # Recommendation details
    recommendation_type = Column(String(100), nullable=False)  # sorting, processing, composting, etc.
    priority_level = Column(String(20), default="medium")  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Implementation details
    estimated_cost_saving = Column(Float)  # Estimated cost saving percentage
    implementation_effort = Column(String(20))  # low, medium, high
    expected_outcome = Column(Text)
    
    # Status tracking
    status = Column(String(50), default="pending")  # pending, implemented, dismissed
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    implemented_at = Column(DateTime)
    
    # Relationships
    analysis_result = relationship("AnalysisResult", backref="recommendations")


class QualityMetric(Base):
    """Model for tracking quality metrics over time."""
    
    __tablename__ = "quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=True)
    
    # Metric details
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    target_value = Column(Float)
    
    # Context
    measurement_context = Column(String(200))  # batch, individual, aggregate
    notes = Column(Text)
    
    # Timestamps
    measured_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("DetectionSession", backref="quality_metrics")
    analysis_result = relationship("AnalysisResult", backref="quality_metrics")


class ProcessingAction(Base):
    """Model for tracking processing actions taken based on recommendations."""
    
    __tablename__ = "processing_actions"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    recommendation_id = Column(Integer, ForeignKey("waste_prevention_recommendations.id"), nullable=True)
    
    # Action details
    action_type = Column(String(100), nullable=False)  # sorting, processing, disposal, etc.
    action_description = Column(Text, nullable=False)
    
    # Quantities
    leaves_processed = Column(Integer)
    estimated_value_recovered = Column(Float)
    waste_amount = Column(Float)
    waste_unit = Column(String(20))  # kg, percentage, etc.
    
    # Results
    actual_outcome = Column(Text)
    success_rating = Column(Integer)  # 1-5 scale
    
    # Timestamps
    planned_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    analysis_result = relationship("AnalysisResult", backref="processing_actions")
    recommendation = relationship("WastePreventionRecommendation", backref="processing_actions")