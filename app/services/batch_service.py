"""Batch processing service for handling multiple images."""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.detection import DetectionSession, DetectionResult, DetectionBox
from app.services.detection_service import TeaLeafDetectionService
from app.database import get_db

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """Service for processing multiple images in batches."""
    
    def __init__(self, db: Session):
        """Initialize batch processing service."""
        self.db = db
        self.detection_service = TeaLeafDetectionService()
        self.upload_dir = "uploads"
        self.results_dir = "results"
        
        # Create directories if they don't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_session(self, name: str, image_count: int) -> DetectionSession:
        """Create a new detection session."""
        
        session = DetectionSession(
            name=name,
            status="pending",
            total_images=image_count,
            processed_images=0
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        logger.info(f"Created detection session {session.id} with {image_count} images")
        return session
    
    def update_session_status(self, session_id: int, status: str, 
                             processed_images: Optional[int] = None):
        """Update session status and progress."""
        
        session = self.db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if session:
            session.status = status
            if processed_images is not None:
                session.processed_images = processed_images
            
            self.db.commit()
            logger.info(f"Updated session {session_id}: status={status}, processed={processed_images}")
    
    def process_batch(self, session_id: int, image_paths: List[str], 
                     confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Process a batch of images for a given session.
        
        Args:
            session_id: ID of the detection session
            image_paths: List of image file paths to process
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Summary of batch processing results
        """
        
        try:
            # Update session status
            self.update_session_status(session_id, "processing")
            
            # Set confidence threshold
            self.detection_service.confidence_threshold = confidence_threshold
            
            # Process images
            results = []
            total_healthy = 0
            total_unhealthy = 0
            total_processing_time = 0.0
            
            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                    
                    # Detect on image
                    detection_result = self.detection_service.detect_image(
                        image_path, 
                        save_annotated=True, 
                        output_dir=self.results_dir
                    )
                    
                    # Create database record
                    db_result = self._create_detection_result(
                        session_id, image_path, detection_result, confidence_threshold
                    )
                    
                    results.append({
                        "id": db_result.id,
                        "image_name": db_result.image_name,
                        "healthy_count": db_result.healthy_count,
                        "unhealthy_count": db_result.unhealthy_count,
                        "health_percentage": db_result.health_percentage,
                        "status": db_result.status
                    })
                    
                    # Update totals
                    total_healthy += db_result.healthy_count
                    total_unhealthy += db_result.unhealthy_count
                    total_processing_time += db_result.processing_time
                    
                    # Update session progress
                    self.update_session_status(session_id, "processing", i + 1)
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    
                    # Create failed result record
                    self._create_failed_result(session_id, image_path, str(e))
                    self.update_session_status(session_id, "processing", i + 1)
            
            # Mark session as completed
            self.update_session_status(session_id, "completed", len(image_paths))
            
            # Calculate summary statistics
            total_images = len(image_paths)
            total_leaves = total_healthy + total_unhealthy
            average_health_percentage = (total_healthy / total_leaves * 100) if total_leaves > 0 else 0.0
            
            summary = {
                "session_id": session_id,
                "total_images": total_images,
                "processed_images": len(results),
                "total_healthy_leaves": total_healthy,
                "total_unhealthy_leaves": total_unhealthy,
                "total_leaves": total_leaves,
                "average_health_percentage": average_health_percentage,
                "total_processing_time": total_processing_time,
                "average_processing_time": total_processing_time / total_images if total_images > 0 else 0.0,
                "status": "completed",
                "results": results
            }
            
            logger.info(f"Batch processing completed for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Batch processing failed for session {session_id}: {e}")
            self.update_session_status(session_id, "failed")
            
            return {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "processed_images": 0,
                "results": []
            }
    
    def _create_detection_result(self, session_id: int, image_path: str, 
                               detection_data: Dict[str, Any], 
                               confidence_threshold: float) -> DetectionResult:
        """Create a DetectionResult database record."""
        
        # Handle error cases
        if "error" in detection_data:
            return self._create_failed_result(session_id, image_path, detection_data["error"])
        
        # Create result record
        result = DetectionResult(
            session_id=session_id,
            image_path=image_path,
            image_name=Path(image_path).name,
            healthy_count=detection_data.get("healthy_count", 0),
            unhealthy_count=detection_data.get("unhealthy_count", 0),
            total_count=detection_data.get("total_count", 0),
            health_percentage=detection_data.get("health_percentage", 0.0),
            confidence_threshold=confidence_threshold,
            processing_time=detection_data.get("processing_time", 0.0),
            annotated_image_path=detection_data.get("annotated_image_path"),
            status="completed"
        )
        
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        
        # Create detection box records
        for box_data in detection_data.get("boxes", []):
            box = DetectionBox(
                result_id=result.id,
                class_id=box_data["class_id"],
                class_name=box_data["class_name"],
                confidence=box_data["confidence"],
                x1=box_data["x1"],
                y1=box_data["y1"],
                x2=box_data["x2"],
                y2=box_data["y2"]
            )
            self.db.add(box)
        
        self.db.commit()
        return result
    
    def _create_failed_result(self, session_id: int, image_path: str, 
                            error_message: str) -> DetectionResult:
        """Create a failed DetectionResult record."""
        
        result = DetectionResult(
            session_id=session_id,
            image_path=image_path,
            image_name=Path(image_path).name,
            healthy_count=0,
            unhealthy_count=0,
            total_count=0,
            health_percentage=0.0,
            confidence_threshold=0.25,
            processing_time=0.0,
            status="failed",
            error_message=error_message
        )
        
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        
        return result
    
    def get_session_progress(self, session_id: int) -> Dict[str, Any]:
        """Get progress information for a session."""
        
        session = self.db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            return {"error": "Session not found"}
        
        percentage = (session.processed_images / session.total_images * 100) if session.total_images > 0 else 0
        
        return {
            "session_id": session.id,
            "name": session.name,
            "status": session.status,
            "processed_images": session.processed_images,
            "total_images": session.total_images,
            "percentage": percentage,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }