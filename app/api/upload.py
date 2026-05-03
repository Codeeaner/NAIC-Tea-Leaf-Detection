"""API endpoints for file upload and detection."""

import os
import shutil
import logging
import json
from typing import List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from test import run_inference
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import (
    DetectionSessionCreateSchema, 
    UploadResponse, 
    DetectionSessionSchema,
    ProgressUpdateSchema
)
from app.services.batch_service import BatchProcessingService
from app.models.detection import DetectionSession, DetectionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# Configuration
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file."""
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        return False
    
    return True


def save_uploaded_file(file: UploadFile, session_id: int) -> str:
    """Save uploaded file to disk and return file path."""
    
    # Create session directory
    session_dir = os.path.join(UPLOAD_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Generate safe filename
    safe_filename = f"{Path(file.filename).stem}_{session_id}{Path(file.filename).suffix}"
    file_path = os.path.join(session_dir, safe_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path


@router.post("/upload/single", response_model=UploadResponse)
async def upload_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_name: str = Form(...),
    confidence_threshold: float = Form(0.25),
    db: Session = Depends(get_db)
):
    """
    Upload and process a single image.
    
    Args:
        file: The image file to upload
        session_name: Name for the detection session
        confidence_threshold: Confidence threshold for detections (0.0-1.0)
        db: Database session
        
    Returns:
        Upload response with session information
    """
    
    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file. Please upload a valid image file (JPG, PNG, WEBP) under 10MB."
            )
        
        # Create detection session
        batch_service = BatchProcessingService(db)
        session = batch_service.create_session(session_name, 1)
        
        # Save uploaded file
        file_path = save_uploaded_file(file, session.id)
        
        # Process image in background
        background_tasks.add_task(
            process_single_image_background,
            session.id,
            file_path,
            confidence_threshold
        )
        
        return UploadResponse(
            session_id=session.id,
            uploaded_files=[file.filename],
            message=f"Image uploaded successfully. Processing started for session {session.id}."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading single image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


@router.post("/upload/batch", response_model=UploadResponse)
async def upload_batch_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_name: str = Form(...),
    confidence_threshold: float = Form(0.25),
    db: Session = Depends(get_db)
):
    """
    Upload and process multiple images.
    
    Args:
        files: List of image files to upload
        session_name: Name for the detection session
        confidence_threshold: Confidence threshold for detections (0.0-1.0)
        db: Database session
        
    Returns:
        Upload response with session information
    """
    
    try:
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
        
        valid_files = []
        invalid_files = []
        
        for file in files:
            if validate_image_file(file):
                valid_files.append(file)
            else:
                invalid_files.append(file.filename)
        
        if not valid_files:
            raise HTTPException(
                status_code=400, 
                detail="No valid image files found. Please upload valid image files (JPG, PNG, WEBP) under 10MB each."
            )
        
        # Create detection session
        batch_service = BatchProcessingService(db)
        session = batch_service.create_session(session_name, len(valid_files))
        
        # Save uploaded files
        file_paths = []
        uploaded_filenames = []
        
        for file in valid_files:
            file_path = save_uploaded_file(file, session.id)
            file_paths.append(file_path)
            uploaded_filenames.append(file.filename)
        
        # Process images in background
        background_tasks.add_task(
            process_batch_images_background,
            session.id,
            file_paths,
            confidence_threshold
        )
        
        message = f"Uploaded {len(valid_files)} files successfully. Processing started for session {session.id}."
        if invalid_files:
            message += f" Skipped {len(invalid_files)} invalid files: {', '.join(invalid_files[:5])}"
            if len(invalid_files) > 5:
                message += f" and {len(invalid_files) - 5} more."
        
        return UploadResponse(
            session_id=session.id,
            uploaded_files=uploaded_filenames,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading batch images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")


def process_single_image_background(
    session_id: int,
    file_path: str,
    confidence_threshold: float
):
    """Background task to process a single image."""
    
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            batch_service = BatchProcessingService(db)
            result = batch_service.process_batch(session_id, [file_path], confidence_threshold)
            logger.info(f"Single image processing completed for session {session_id}")
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error processing single image for session {session_id}: {e}")
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            batch_service = BatchProcessingService(db)
            batch_service.update_session_status(session_id, "failed")
        finally:
            db.close()


def process_batch_images_background(
    session_id: int,
    file_paths: List[str],
    confidence_threshold: float
):
    """Background task to process multiple images."""
    
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            batch_service = BatchProcessingService(db)
            result = batch_service.process_batch(session_id, file_paths, confidence_threshold)
            logger.info(f"Batch processing completed for session {session_id}")
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error processing batch for session {session_id}: {e}")
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            batch_service = BatchProcessingService(db)
            batch_service.update_session_status(session_id, "failed")
        finally:
            db.close()


@router.get("/upload/progress/{session_id}", response_model=ProgressUpdateSchema)
async def get_upload_progress(session_id: int, db: Session = Depends(get_db)):
    """
    Get processing progress for a session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Progress information
    """
    
    try:
        batch_service = BatchProcessingService(db)
        progress = batch_service.get_session_progress(session_id)
        
        if "error" in progress:
            raise HTTPException(status_code=404, detail=progress["error"])
        
        return ProgressUpdateSchema(
            session_id=progress["session_id"],
            processed_images=progress["processed_images"],
            total_images=progress["total_images"],
            current_image="Processing...",
            status=progress["status"],
            percentage=progress["percentage"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/upload/session/{session_id}")
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    """
    Delete a detection session and associated files.
    
    Args:
        session_id: ID of the detection session to delete
        db: Database session
        
    Returns:
        Success message
    """
    
    try:
        # Get session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete session files
        session_dir = os.path.join(UPLOAD_DIR, f"session_{session_id}")
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        
        # Delete from database
        db.delete(session)
        db.commit()
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/upload/sessions/all")
async def delete_all_sessions(db: Session = Depends(get_db)):
    """
    Delete all detection sessions and associated files.
    
    Args:
        db: Database session
        
    Returns:
        Success message with count of deleted sessions
    """
    
    try:
        # Get all sessions
        sessions = db.query(DetectionSession).all()
        deleted_count = len(sessions)
        
        if deleted_count == 0:
            return {"message": "No sessions to delete", "deleted_count": 0}
        
        # Delete associated files and database records
        for session in sessions:
            try:
                # Delete session directory and files
                session_dir = os.path.join("uploads", f"session_{session.id}")
                if os.path.exists(session_dir):
                    import shutil
                    shutil.rmtree(session_dir)
                    logger.info(f"Deleted session directory: {session_dir}")
                
                # Delete associated detection results
                db.query(DetectionResult).filter(DetectionResult.session_id == session.id).delete()
                
                # Delete associated analytics results
                from app.models.analytics import AnalysisResult, BatchAnalysisResult
                db.query(AnalysisResult).filter(AnalysisResult.session_id == session.id).delete()
                db.query(BatchAnalysisResult).filter(BatchAnalysisResult.session_id == session.id).delete()
                
                # Delete analytics files
                analytics_dir = "analytics"
                if os.path.exists(analytics_dir):
                    for filename in os.listdir(analytics_dir):
                        if filename.endswith('.json'):
                            try:
                                filepath = os.path.join(analytics_dir, filename)
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    if data.get('session_id') == session.id:
                                        os.remove(filepath)
                                        logger.info(f"Deleted analytics file: {filepath}")
                            except Exception as e:
                                logger.warning(f"Could not check/delete analytics file {filename}: {e}")
                
            except Exception as e:
                logger.error(f"Error deleting files for session {session.id}: {e}")
                # Continue with other sessions
        
        # Delete all sessions from database
        db.query(DetectionSession).delete()
        db.commit()
        
        logger.info(f"Successfully deleted {deleted_count} sessions and associated data")
        return {"message": f"Successfully deleted {deleted_count} sessions and all associated data", "deleted_count": deleted_count}
        
    except Exception as e:
        logger.error(f"Error deleting all sessions: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
@router.post("/detect", response_model=dict)
async def detect_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Detect tea leaf health status using the YOLO model.
    
    Args:
        file: The image file to detect
        db: Database session
        
    Returns:
        Detection results
    """
    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file. Please upload a valid image file (JPG, PNG, WEBP) under 10MB."
            )
        
        # Save uploaded file temporarily
        temp_file_path = save_uploaded_file(file, session_id=0)  # Use session_id=0 for temporary
        
        # Run inference using test.py
        from test import run_inference
        results = run_inference(source=temp_file_path, save=False, show=False)
        
        # Parse results
        detection_summary = {
            "healthy_leaves": sum(1 for r in results if r.boxes and any(int(b.cls[0]) == 1 for b in r.boxes)),
            "unhealthy_leaves": sum(1 for r in results if r.boxes and any(int(b.cls[0]) == 0 for b in r.boxes)),
            "total_leaves": sum(1 for r in results if r.boxes)
        }
        
        return detection_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during detection")
