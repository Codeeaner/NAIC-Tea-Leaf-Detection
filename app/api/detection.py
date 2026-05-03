"""API endpoints for detection results and history."""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from app.database import get_db
from app.schemas import (
    DetectionSessionSchema,
    DetectionSessionSummarySchema,
    DetectionResultSchema,
    DetectionStatsSchema
)
from app.models.detection import DetectionSession, DetectionResult
from datetime import datetime, date

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detection"])


@router.get("/results/{session_id}", response_model=DetectionSessionSchema)
async def get_detection_results(session_id: int, db: Session = Depends(get_db)):
    """
    Get detailed detection results for a specific session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Complete session data with results and bounding boxes
    """
    
    try:
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions", response_model=List[DetectionSessionSummarySchema])
async def get_detection_sessions(
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of sessions to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """
    Get list of detection sessions with summary information.
    
    Args:
        skip: Number of sessions to skip for pagination
        limit: Maximum number of sessions to return
        status: Optional status filter (pending, processing, completed, failed)
        db: Database session
        
    Returns:
        List of session summaries
    """
    
    try:
        query = db.query(DetectionSession)
        
        # Apply status filter if provided
        if status:
            query = query.filter(DetectionSession.status == status)
        
        # Order by creation date (newest first)
        query = query.order_by(desc(DetectionSession.created_at))
        
        # Apply pagination
        sessions = query.offset(skip).limit(limit).all()
        
        # Calculate summary statistics for each session
        session_summaries = []
        for session in sessions:
            completed_results = [r for r in session.results if r.status == "completed"]
            
            total_healthy = sum(r.healthy_count for r in completed_results)
            total_unhealthy = sum(r.unhealthy_count for r in completed_results)
            total_leaves = total_healthy + total_unhealthy
            
            avg_health_percentage = (
                sum(r.health_percentage for r in completed_results) / len(completed_results)
                if completed_results else 0.0
            )
            
            summary = DetectionSessionSummarySchema(
                id=session.id,
                name=session.name,
                status=session.status,
                total_images=session.total_images,
                processed_images=session.processed_images,
                total_healthy=total_healthy,
                total_unhealthy=total_unhealthy,
                average_health_percentage=avg_health_percentage,
                created_at=session.created_at,
                updated_at=session.updated_at
            )
            
            session_summaries.append(summary)
        
        return session_summaries
        
    except Exception as e:
        logger.error(f"Error getting detection sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=List[DetectionSessionSummarySchema])
async def get_detection_history(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of sessions"),
    db: Session = Depends(get_db)
):
    """
    Get detection history for the specified number of days.
    
    Args:
        days: Number of days to look back
        limit: Maximum number of sessions to return
        db: Database session
        
    Returns:
        List of recent session summaries
    """
    
    try:
        # Calculate date threshold
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        
        # Query sessions from the last N days
        query = db.query(DetectionSession).filter(
            DetectionSession.created_at >= cutoff_date
        ).order_by(desc(DetectionSession.created_at)).limit(limit)
        
        sessions = query.all()
        
        # Calculate summary statistics
        session_summaries = []
        for session in sessions:
            completed_results = [r for r in session.results if r.status == "completed"]
            
            total_healthy = sum(r.healthy_count for r in completed_results)
            total_unhealthy = sum(r.unhealthy_count for r in completed_results)
            
            avg_health_percentage = (
                sum(r.health_percentage for r in completed_results) / len(completed_results)
                if completed_results else 0.0
            )
            
            summary = DetectionSessionSummarySchema(
                id=session.id,
                name=session.name,
                status=session.status,
                total_images=session.total_images,
                processed_images=session.processed_images,
                total_healthy=total_healthy,
                total_unhealthy=total_unhealthy,
                average_health_percentage=avg_health_percentage,
                created_at=session.created_at,
                updated_at=session.updated_at
            )
            
            session_summaries.append(summary)
        
        return session_summaries
        
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=DetectionStatsSchema)
async def get_detection_statistics(db: Session = Depends(get_db)):
    """
    Get overall detection statistics.
    
    Args:
        db: Database session
        
    Returns:
        Overall statistics including total sessions, images, and health data
    """
    
    try:
        # Get total counts
        total_sessions = db.query(DetectionSession).count()
        
        # Get completed results only for accurate statistics
        completed_results = db.query(DetectionResult).filter(
            DetectionResult.status == "completed"
        ).all()
        
        total_images_processed = len(completed_results)
        total_healthy_leaves = sum(r.healthy_count for r in completed_results)
        total_unhealthy_leaves = sum(r.unhealthy_count for r in completed_results)
        
        # Calculate average health percentage
        total_leaves = total_healthy_leaves + total_unhealthy_leaves
        average_health_percentage = (
            (total_healthy_leaves / total_leaves * 100) if total_leaves > 0 else 0.0
        )
        
        # Get today's statistics
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time())
        
        sessions_today = db.query(DetectionSession).filter(
            DetectionSession.created_at >= today_start
        ).count()
        
        images_today = db.query(DetectionResult).filter(
            DetectionResult.created_at >= today_start,
            DetectionResult.status == "completed"
        ).count()
        
        return DetectionStatsSchema(
            total_sessions=total_sessions,
            total_images_processed=total_images_processed,
            total_healthy_leaves=total_healthy_leaves,
            total_unhealthy_leaves=total_unhealthy_leaves,
            average_health_percentage=average_health_percentage,
            sessions_today=sessions_today,
            images_processed_today=images_today
        )
        
    except Exception as e:
        logger.error(f"Error getting detection statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}/results", response_model=List[DetectionResultSchema])
async def get_session_results_only(
    session_id: int,
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of results to return"),
    db: Session = Depends(get_db)
):
    """
    Get detection results for a session without full session data.
    
    Args:
        session_id: ID of the detection session
        skip: Number of results to skip for pagination
        limit: Maximum number of results to return
        db: Database session
        
    Returns:
        List of detection results
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get results with pagination
        results = db.query(DetectionResult).filter(
            DetectionResult.session_id == session_id
        ).order_by(DetectionResult.created_at).offset(skip).limit(limit).all()
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: int, db: Session = Depends(get_db)):
    """
    Get a quick summary of session results.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Session summary with key statistics
    """
    
    try:
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate statistics
        completed_results = [r for r in session.results if r.status == "completed"]
        failed_results = [r for r in session.results if r.status == "failed"]
        
        total_healthy = sum(r.healthy_count for r in completed_results)
        total_unhealthy = sum(r.unhealthy_count for r in completed_results)
        total_leaves = total_healthy + total_unhealthy
        
        avg_health_percentage = (
            sum(r.health_percentage for r in completed_results) / len(completed_results)
            if completed_results else 0.0
        )
        
        avg_processing_time = (
            sum(r.processing_time for r in completed_results) / len(completed_results)
            if completed_results else 0.0
        )
        
        return {
            "session_id": session.id,
            "session_name": session.name,
            "status": session.status,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "total_images": session.total_images,
            "processed_images": session.processed_images,
            "completed_images": len(completed_results),
            "failed_images": len(failed_results),
            "total_healthy_leaves": total_healthy,
            "total_unhealthy_leaves": total_unhealthy,
            "total_leaves_detected": total_leaves,
            "average_health_percentage": round(avg_health_percentage, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "completion_percentage": round((session.processed_images / session.total_images * 100) if session.total_images > 0 else 0, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")