"""API endpoints for report generation and download."""

import os
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.report_service import ReportService
from app.models.detection import DetectionSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/reports/generate/{session_id}")
async def generate_reports(
    session_id: int,
    background_tasks: BackgroundTasks,
    report_types: str = "pdf,csv",  # Comma-separated list: pdf, csv, or both
    db: Session = Depends(get_db)
):
    """
    Generate reports for a detection session.
    
    Args:
        session_id: ID of the detection session
        report_types: Types of reports to generate (pdf, csv, or both)
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Confirmation message with generation status
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail="Reports can only be generated for completed sessions"
            )
        
        # Parse report types
        requested_types = [t.strip().lower() for t in report_types.split(",")]
        valid_types = ["pdf", "csv"]
        invalid_types = [t for t in requested_types if t not in valid_types]
        
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid report types: {', '.join(invalid_types)}. Valid types: {', '.join(valid_types)}"
            )
        
        # Generate reports in background
        background_tasks.add_task(
            generate_reports_background,
            session_id,
            requested_types,
            db
        )
        
        return {
            "message": f"Report generation started for session {session_id}",
            "session_id": session_id,
            "report_types": requested_types,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting report generation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/download/pdf/{session_id}")
async def download_pdf_report(session_id: int, db: Session = Depends(get_db)):
    """
    Download PDF report for a session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        PDF file download
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate PDF report
        report_service = ReportService(db)
        pdf_path = report_service.generate_pdf_report(session_id)
        
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate PDF report"
            )
        
        # Return file download
        filename = f"tea_leaf_report_session_{session_id}.pdf"
        return FileResponse(
            path=pdf_path,
            filename=filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF report for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/download/csv/{session_id}")
async def download_csv_report(session_id: int, db: Session = Depends(get_db)):
    """
    Download CSV report for a session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        CSV file download
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate CSV report
        report_service = ReportService(db)
        csv_path = report_service.generate_csv_report(session_id)
        
        if not csv_path or not os.path.exists(csv_path):
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate CSV report"
            )
        
        # Return file download
        filename = f"tea_leaf_data_session_{session_id}.csv"
        return FileResponse(
            path=csv_path,
            filename=filename,
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV report for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/status/{session_id}")
async def get_report_status(session_id: int, db: Session = Depends(get_db)):
    """
    Get available reports for a session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Available report information
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check for existing report files
        reports_dir = "reports"
        available_reports = []
        
        # Check for PDF reports
        pdf_pattern = f"tea_leaf_report_session_{session_id}_"
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.startswith(pdf_pattern) and filename.endswith(".pdf"):
                    file_path = os.path.join(reports_dir, filename)
                    file_stats = os.stat(file_path)
                    available_reports.append({
                        "type": "pdf",
                        "filename": filename,
                        "size": file_stats.st_size,
                        "created": file_stats.st_mtime
                    })
        
        # Check for CSV reports
        csv_pattern = f"tea_leaf_data_session_{session_id}_"
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.startswith(csv_pattern) and filename.endswith(".csv"):
                    file_path = os.path.join(reports_dir, filename)
                    file_stats = os.stat(file_path)
                    available_reports.append({
                        "type": "csv",
                        "filename": filename,
                        "size": file_stats.st_size,
                        "created": file_stats.st_mtime
                    })
        
        return {
            "session_id": session_id,
            "session_name": session.name,
            "session_status": session.status,
            "can_generate_reports": session.status == "completed",
            "available_reports": available_reports,
            "total_reports": len(available_reports)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/summary/{session_id}")
async def get_report_summary(session_id: int, db: Session = Depends(get_db)):
    """
    Get a summary for report generation without generating the actual report.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Report summary data
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get summary from report service
        report_service = ReportService(db)
        summary = report_service.get_session_summary(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def generate_reports_background(
    session_id: int, 
    report_types: list, 
    db: Session
):
    """Background task to generate reports."""
    
    try:
        report_service = ReportService(db)
        
        generated_reports = []
        
        if "pdf" in report_types:
            pdf_path = report_service.generate_pdf_report(session_id)
            if pdf_path:
                generated_reports.append(f"PDF: {pdf_path}")
                logger.info(f"PDF report generated for session {session_id}: {pdf_path}")
        
        if "csv" in report_types:
            csv_path = report_service.generate_csv_report(session_id)
            if csv_path:
                generated_reports.append(f"CSV: {csv_path}")
                logger.info(f"CSV report generated for session {session_id}: {csv_path}")
        
        if generated_reports:
            logger.info(f"Reports generated for session {session_id}: {', '.join(generated_reports)}")
        else:
            logger.warning(f"No reports generated for session {session_id}")
            
    except Exception as e:
        logger.error(f"Error generating reports for session {session_id}: {e}")


@router.delete("/reports/cleanup/{session_id}")
async def cleanup_session_reports(session_id: int, db: Session = Depends(get_db)):
    """
    Clean up generated report files for a session.
    
    Args:
        session_id: ID of the detection session
        db: Database session
        
    Returns:
        Cleanup status
    """
    
    try:
        # Verify session exists
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Clean up report files
        reports_dir = "reports"
        cleaned_files = []
        
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if f"session_{session_id}_" in filename:
                    file_path = os.path.join(reports_dir, filename)
                    try:
                        os.remove(file_path)
                        cleaned_files.append(filename)
                    except Exception as e:
                        logger.warning(f"Failed to remove file {file_path}: {e}")
        
        return {
            "message": f"Cleaned up reports for session {session_id}",
            "session_id": session_id,
            "cleaned_files": cleaned_files,
            "total_cleaned": len(cleaned_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up reports for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")