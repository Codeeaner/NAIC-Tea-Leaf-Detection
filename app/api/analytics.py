import os
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.analytics_service import AnalyticsService
from app.services.decision_support_service import DecisionSupportService
from app.models.detection import DetectionSession, DetectionResult
from app.models.analytics import (
    AnalysisResult, 
    BatchAnalysisResult, 
    WastePreventionRecommendation,
    QualityMetric,
    ProcessingAction
)
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


def _build_decision_support_payload(
    session: DetectionSession,
    db: Session,
    lot_value_estimate: Optional[float] = None,
    estimated_weight_kg: Optional[float] = None,
    price_per_kg: Optional[float] = None,
    currency: Optional[str] = None,
) -> dict:
    """Build a decision-support payload from completed results in a session."""

    completed_results = db.query(DetectionResult).filter(
        DetectionResult.session_id == session.id,
        DetectionResult.status == "completed"
    ).all()

    healthy_count = sum(result.healthy_count for result in completed_results)
    unhealthy_count = sum(result.unhealthy_count for result in completed_results)

    decision_service = DecisionSupportService()
    return decision_service.build_decision_support(
        session_id=session.id,
        session_name=session.name,
        total_images=session.total_images,
        completed_images=len(completed_results),
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        lot_value_estimate=lot_value_estimate,
        estimated_weight_kg=estimated_weight_kg,
        price_per_kg=price_per_kg,
        currency=currency,
    )


def _build_dashboard_payload(
    session: DetectionSession,
    individual_analyses: List[AnalysisResult],
    decision_support: dict,
    batch_analysis: Optional[BatchAnalysisResult] = None,
) -> dict:
    """Build a dashboard payload for trend and impact visualizations."""

    def _label_for_analysis(index: int, created_at_value) -> str:
        if isinstance(created_at_value, datetime):
            return created_at_value.strftime("%d %b")

        if created_at_value:
            try:
                return datetime.fromisoformat(str(created_at_value)).strftime("%d %b")
            except (TypeError, ValueError):
                pass

        return f"Scan {index + 1}"

    sorted_analyses = sorted(
        individual_analyses,
        key=lambda analysis: analysis.created_at.isoformat() if analysis.created_at else "",
    )

    summary = decision_support.get("summary", {}) if isinstance(decision_support, dict) else {}
    batch_decision = decision_support.get("batch_decision", {}) if isinstance(decision_support, dict) else {}
    economic_impact = decision_support.get("economic_impact", {}) if isinstance(decision_support, dict) else {}
    detection_summary = decision_support.get("detection_summary", {}) if isinstance(decision_support, dict) else {}

    health_percentage = float(summary.get("health_percentage") or 0.0)
    healthy_percentage = round(health_percentage, 1)
    unhealthy_percentage = round(max(100.0 - health_percentage, 0.0), 1)
    healthy_count = int(detection_summary.get("healthy_count") or 0)
    unhealthy_count = int(detection_summary.get("unhealthy_count") or 0)

    quality_trend = []
    for index, analysis in enumerate(sorted_analyses):
        quality_trend.append({
            "label": _label_for_analysis(index, analysis.created_at),
            "health_percentage": round(float(analysis.health_percentage or 0.0), 1),
            "status": analysis.status,
        })

    if not quality_trend and batch_analysis:
        quality_trend.append({
            "label": "Current batch",
            "health_percentage": round(float(batch_analysis.overall_health_percentage or 0.0), 1),
            "status": batch_analysis.status,
        })

    if not quality_trend:
        quality_trend.append({
            "label": "Current batch",
            "health_percentage": healthy_percentage,
            "status": decision_support.get("status", "completed") if isinstance(decision_support, dict) else "completed",
        })

    currency = economic_impact.get("currency", "USD")
    money_saved = float(economic_impact.get("profit_improvement_per_batch") or economic_impact.get("potential_savings_from_waste_reduction") or 0.0)
    waste_reduced_kg = float(economic_impact.get("waste_reduction_kg") or 0.0)

    return {
        "summary": {
            "session_name": session.name,
            "total_scans": len(individual_analyses),
            "health_percentage": healthy_percentage,
            "healthy_percentage": healthy_percentage,
            "unhealthy_percentage": unhealthy_percentage,
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "waste_reduced_kg": waste_reduced_kg,
            "money_saved": money_saved,
            "currency": currency,
            "recommended_action": batch_decision.get("action_label") if isinstance(batch_decision, dict) else None,
            "quality_grade": economic_impact.get("sale_grade") or summary.get("quality_grade"),
        },
        "quality_trend": quality_trend,
        "composition": {
            "healthy_leaves": healthy_count,
            "unhealthy_leaves": unhealthy_count,
            "healthy_percentage": healthy_percentage,
            "unhealthy_percentage": unhealthy_percentage,
        },
        "economic": {
            "currency": currency,
            "estimated_income_loss": float(economic_impact.get("estimated_income_loss") or 0.0),
            "potential_savings_from_waste_reduction": float(economic_impact.get("potential_savings_from_waste_reduction") or 0.0),
            "profit_before_action": float(economic_impact.get("profit_before_action") or 0.0),
            "profit_after_action": float(economic_impact.get("profit_after_action") or 0.0),
            "profit_improvement_per_batch": float(economic_impact.get("profit_improvement_per_batch") or 0.0),
            "profit_improvement_pct": float(economic_impact.get("profit_improvement_pct") or 0.0),
        },
        "waste": {
            "waste_before_kg": float(economic_impact.get("waste_before_kg") or 0.0),
            "waste_after_kg": float(economic_impact.get("waste_after_kg") or 0.0),
            "waste_reduction_kg": waste_reduced_kg,
            "waste_reduction_pct": float(economic_impact.get("waste_reduction_pct") or 0.0),
            "recovered_value": float(economic_impact.get("recovered_value") or 0.0),
        },
        "decision": {
            "action": batch_decision.get("action") if isinstance(batch_decision, dict) else None,
            "action_label": batch_decision.get("action_label") if isinstance(batch_decision, dict) else None,
            "reason": batch_decision.get("reason") if isinstance(batch_decision, dict) else None,
            "allocation": batch_decision.get("allocation") if isinstance(batch_decision, dict) else {},
        },
    }


@router.post("/analyze/{result_id}")
async def analyze_detection_result(
    result_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze a single detection result using Qwen3-VL.
    
    Args:
        result_id: ID of the detection result to analyze
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Analysis initiation response
    """
    
    try:
        # Get detection result
        detection_result = db.query(DetectionResult).filter(
            DetectionResult.id == result_id
        ).first()
        
        if not detection_result:
            raise HTTPException(status_code=404, detail="Detection result not found")
        
        if not detection_result.annotated_image_path or not os.path.exists(detection_result.annotated_image_path):
            raise HTTPException(
                status_code=400, 
                detail="Annotated image not found for analysis"
            )
        
        # Check if analysis already exists
        existing_analysis = db.query(AnalysisResult).filter(
            AnalysisResult.detection_result_id == result_id
        ).first()
        
        if existing_analysis:
            return {
                "message": "Analysis already exists for this detection result",
                "analysis_id": existing_analysis.analysis_id,
                "status": existing_analysis.status
            }
        
        # Start analysis in background
        background_tasks.add_task(
            analyze_detection_background,
            result_id,
            db
        )
        
        return {
            "message": f"Analysis started for detection result {result_id}",
            "result_id": result_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for detection result {result_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch")
async def analyze_batch_results(
    session_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze all detection results in a session using Qwen3-VL.
    
    Args:
        session_id: ID of the detection session
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Batch analysis initiation response
    """
    
    try:
        # Get detection session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Can only analyze completed sessions"
            )
        
        # Check if batch analysis already exists
        existing_batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == session_id
        ).first()
        
        if existing_batch_analysis:
            return {
                "message": "Batch analysis already exists for this session",
                "batch_analysis_id": existing_batch_analysis.batch_analysis_id,
                "status": existing_batch_analysis.status
            }
        
        # Generate batch analysis ID
        timestamp = int(datetime.now().timestamp())
        unique_id = uuid.uuid4().hex[:8]
        batch_analysis_id = f"batch_{unique_id}_{timestamp}"
        
        # Get detection results count for initial record
        results_count = db.query(DetectionResult).filter(
            DetectionResult.session_id == session_id,
            DetectionResult.status == "completed"
        ).count()
        
        # Create batch analysis record with processing status
        batch_analysis_record = BatchAnalysisResult(
            batch_analysis_id=batch_analysis_id,
            session_id=session_id,
            total_images=results_count,
            analyzed_images=0,
            processing_time=0.0,
            status="processing",
            total_healthy_leaves=0,
            total_unhealthy_leaves=0,
            total_leaves=0,
            overall_health_percentage=0.0,
            aggregate_recommendations={},
            individual_analysis_ids=[]
        )
        
        db.add(batch_analysis_record)
        db.commit()
        db.refresh(batch_analysis_record)
        
        # Start batch analysis in background
        background_tasks.add_task(
            analyze_batch_background,
            session_id,
            batch_analysis_id,
            db
        )
        
        return {
            "message": f"Batch analysis started for session {session_id}",
            "batch_analysis_id": batch_analysis_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch/{batch_id}")
async def analyze_batch_results_by_id(
    batch_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze all detection results in a batch using Qwen3-VL, identified by batch ID.
    
    Args:
        batch_id: ID of the batch (session ID)
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Batch analysis initiation response
    """
    
    try:
        # Get detection session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == batch_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Batch session not found")
        
        if session.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Can only analyze completed batch sessions"
            )
        
        # Check if batch analysis already exists
        existing_batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == batch_id
        ).first()
        
        if existing_batch_analysis:
            return {
                "message": "Batch analysis already exists for this batch",
                "batch_analysis_id": existing_batch_analysis.batch_analysis_id,
                "status": existing_batch_analysis.status
            }
        
        # Generate batch analysis ID
        timestamp = int(datetime.now().timestamp())
        unique_id = uuid.uuid4().hex[:8]
        batch_analysis_id = f"batch_{unique_id}_{timestamp}"
        
        # Get detection results count for initial record
        results_count = db.query(DetectionResult).filter(
            DetectionResult.session_id == batch_id,
            DetectionResult.status == "completed"
        ).count()
        
        # Create batch analysis record with processing status
        batch_analysis_record = BatchAnalysisResult(
            batch_analysis_id=batch_analysis_id,
            session_id=batch_id,
            total_images=results_count,
            analyzed_images=0,
            processing_time=0.0,
            status="processing",
            total_healthy_leaves=0,
            total_unhealthy_leaves=0,
            total_leaves=0,
            overall_health_percentage=0.0,
            aggregate_recommendations={},
            individual_analysis_ids=[]
        )
        
        db.add(batch_analysis_record)
        db.commit()
        db.refresh(batch_analysis_record)
        
        # Start batch analysis in background
        background_tasks.add_task(
            analyze_batch_background,
            batch_id,
            batch_analysis_id,
            db
        )
        
        return {
            "message": f"Batch analysis started for batch {batch_id}",
            "batch_analysis_id": batch_analysis_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis for batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Get analysis result by analysis ID.
    
    Args:
        analysis_id: Analysis ID
        db: Database session
        
    Returns:
        Analysis result
    """
    
    try:
        analysis_result = db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis_id
        ).first()
        
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        logger.info(f"Retrieved analysis result for analysis_id: {analysis_id}")
        return {
            "analysis_id": analysis_result.analysis_id,
            "status": analysis_result.status,
            "model_used": analysis_result.model_used,
            "processing_time": analysis_result.processing_time,
            "detection_summary": {
                "healthy_count": analysis_result.healthy_count,
                "unhealthy_count": analysis_result.unhealthy_count,
                "total_count": analysis_result.total_count,
                "health_percentage": analysis_result.health_percentage
            },
            "ai_analysis": analysis_result.ai_analysis,
            "error_message": analysis_result.error_message,
            "created_at": analysis_result.created_at,
            "image_path": analysis_result.image_path,
            "annotated_image_path": analysis_result.annotated_image_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis result {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/batch/{batch_analysis_id}")
async def get_batch_analysis_result(
    batch_analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Get batch analysis result.
    
    Args:
        batch_analysis_id: Batch analysis ID (can be UUID string or session ID)
        db: Database session
        
    Returns:
        Batch analysis result
    """
    
    try:
        batch_result = None
        # First, try to find by batch_analysis_id (UUID string)
        batch_result = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.batch_analysis_id == batch_analysis_id
        ).first()

        # If not found, and the input looks like an integer, try to find by session_id
        if not batch_result:
            try:
                session_id_as_int = int(batch_analysis_id)
                batch_result = db.query(BatchAnalysisResult).filter(
                    BatchAnalysisResult.session_id == session_id_as_int
                ).first()
            except ValueError:
                # The batch_analysis_id was not an integer, so it's not a session ID
                pass
        
        if not batch_result:
            raise HTTPException(status_code=404, detail="Batch analysis result not found")
        
        return {
            "batch_analysis_id": batch_result.batch_analysis_id,
            "session_id": batch_result.session_id,
            "status": batch_result.status,
            "total_images": batch_result.total_images,
            "analyzed_images": batch_result.analyzed_images,
            "processing_time": batch_result.processing_time,
            "batch_summary": {
                "total_healthy_leaves": batch_result.total_healthy_leaves,
                "total_unhealthy_leaves": batch_result.total_unhealthy_leaves,
                "total_leaves": batch_result.total_leaves,
                "overall_health_percentage": batch_result.overall_health_percentage
            },
            "aggregate_recommendations": batch_result.aggregate_recommendations,
            "individual_analysis_ids": batch_result.individual_analysis_ids,
            "created_at": batch_result.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch analysis result {batch_analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}")
async def get_session_analytics(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all analytics for a session.
    
    Args:
        session_id: Session ID
        db: Database session
        
    Returns:
        Session analytics summary
    """
    
    try:
        # Get session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get individual analyses
        individual_analyses = db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session_id
        ).all()
        
        # Get batch analysis
        batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == session_id
        ).first()
        
        # Get recommendations
        recommendations = []
        for analysis in individual_analyses:
            analysis_recommendations = db.query(WastePreventionRecommendation).filter(
                WastePreventionRecommendation.analysis_result_id == analysis.id
            ).all()
            recommendations.extend(analysis_recommendations)

        decision_support = _build_decision_support_payload(session, db)
        dashboard = _build_dashboard_payload(session, individual_analyses, decision_support, batch_analysis)
        
        return {
            "session_id": session_id,
            "session_name": session.name,
            "individual_analyses": [
                {
                    "analysis_id": analysis.analysis_id,
                    "status": analysis.status,
                    "health_percentage": analysis.health_percentage,
                    "created_at": analysis.created_at
                } for analysis in individual_analyses
            ],
            "batch_analysis": {
                "batch_analysis_id": str(batch_analysis.batch_analysis_id) if batch_analysis and isinstance(batch_analysis.batch_analysis_id, (str, int)) else None,
                "status": batch_analysis.status,
                "overall_health_percentage": batch_analysis.overall_health_percentage,
                "created_at": batch_analysis.created_at
            } if batch_analysis else None,
            "decision_support": decision_support,
            "dashboard": dashboard,
            "total_recommendations": len(recommendations),
            "pending_recommendations": len([r for r in recommendations if r.status == "pending"]),
            "implemented_recommendations": len([r for r in recommendations if r.status == "implemented"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session analytics {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/decision-support/{session_id}")
async def get_decision_support(
    session_id: int,
    lot_value_estimate: Optional[float] = Query(None, ge=0, description="Optional estimated batch value"),
    estimated_weight_kg: Optional[float] = Query(None, ge=0, description="Optional estimated batch weight in kilograms"),
    price_per_kg: Optional[float] = Query(None, ge=0, description="Optional market price per kilogram"),
    currency: Optional[str] = Query(None, min_length=3, max_length=3, description="Optional 3-letter currency code"),
    db: Session = Depends(get_db)
):
    """Return farmer-facing batch decision support for a session."""

    try:
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return _build_decision_support_payload(
            session,
            db,
            lot_value_estimate=lot_value_estimate,
            estimated_weight_kg=estimated_weight_kg,
            price_per_kg=price_per_kg,
            currency=currency,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building decision support for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations/{analysis_id}")
async def get_recommendations(
    analysis_id: str,
    priority: Optional[str] = Query(None, description="Filter by priority level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """
    Get waste prevention recommendations for an analysis.
    
    Args:
        analysis_id: Analysis ID
        priority: Filter by priority level
        status: Filter by status
        db: Database session
        
    Returns:
        List of recommendations
    """
    
    try:
        # Get analysis result
        analysis_result = db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis_id
        ).first()
        
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        # Build query
        query = db.query(WastePreventionRecommendation).filter(
            WastePreventionRecommendation.analysis_result_id == analysis_result.id
        )
        
        if priority:
            query = query.filter(WastePreventionRecommendation.priority_level == priority)
        
        if status:
            query = query.filter(WastePreventionRecommendation.status == status)
        
        recommendations = query.all()
        
        return [
            {
                "id": rec.id,
                "recommendation_type": rec.recommendation_type,
                "priority_level": rec.priority_level,
                "title": rec.title,
                "description": rec.description,
                "estimated_cost_saving": rec.estimated_cost_saving,
                "implementation_effort": rec.implementation_effort,
                "expected_outcome": rec.expected_outcome,
                "status": rec.status,
                "notes": rec.notes,
                "created_at": rec.created_at,
                "implemented_at": rec.implemented_at
            } for rec in recommendations
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history")
async def get_analytics_history(
    limit: int = Query(20, description="Maximum number of results"),
    session_id: Optional[int] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db)
):
    """
    Get analytics history.
    
    Args:
        limit: Maximum number of results
        session_id: Filter by session ID
        db: Database session
        
    Returns:
        List of analytics results
    """
    
    try:
        # Build query for individual analyses
        query = db.query(AnalysisResult)
        
        if session_id:
            query = query.filter(AnalysisResult.session_id == session_id)
        
        analyses = query.order_by(AnalysisResult.created_at.desc()).limit(limit).all()
        
        # Build query for batch analyses
        batch_query = db.query(BatchAnalysisResult)
        
        if session_id:
            batch_query = batch_query.filter(BatchAnalysisResult.session_id == session_id)
        
        batch_analyses = batch_query.order_by(BatchAnalysisResult.created_at.desc()).limit(limit).all()
        
        return {
            "individual_analyses": [
                {
                    "analysis_id": analysis.analysis_id,
                    "session_id": analysis.session_id,
                    "status": analysis.status,
                    "model_used": analysis.model_used,
                    "health_percentage": analysis.health_percentage,
                    "processing_time": analysis.processing_time,
                    "created_at": analysis.created_at
                } for analysis in analyses
            ],
            "batch_analyses": [
                {
                    "batch_analysis_id": batch.batch_analysis_id,
                    "session_id": batch.session_id,
                    "status": batch.status,
                    "total_images": batch.total_images,
                    "analyzed_images": batch.analyzed_images,
                    "overall_health_percentage": batch.overall_health_percentage,
                    "processing_time": batch.processing_time,
                    "created_at": batch.created_at
                } for batch in batch_analyses
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/recommendations/{recommendation_id}/implement")
async def implement_recommendation(
    recommendation_id: int,
    notes: str = "",
    db: Session = Depends(get_db)
):
    """
    Mark a recommendation as implemented.
    
    Args:
        recommendation_id: Recommendation ID
        notes: Implementation notes
        db: Database session
        
    Returns:
        Updated recommendation
    """
    
    try:
        recommendation = db.query(WastePreventionRecommendation).filter(
            WastePreventionRecommendation.id == recommendation_id
        ).first()
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        recommendation.status = "implemented"
        recommendation.notes = notes
        recommendation.implemented_at = datetime.utcnow()
        
        db.commit()
        db.refresh(recommendation)
        
        return {
            "message": "Recommendation marked as implemented",
            "recommendation_id": recommendation_id,
            "status": recommendation.status,
            "implemented_at": recommendation.implemented_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error implementing recommendation {recommendation_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


def analyze_detection_background(result_id: int, db: Session):
    """Background task to analyze a detection result."""
    
    try:
        analytics_service = AnalyticsService()
        
        # Get detection result
        detection_result = db.query(DetectionResult).filter(
            DetectionResult.id == result_id
        ).first()
        
        if not detection_result:
            logger.error(f"Detection result {result_id} not found")
            return
        
        # Prepare detection data
        detection_data = {
            "healthy_count": detection_result.healthy_count,
            "unhealthy_count": detection_result.unhealthy_count,
            "total_count": detection_result.total_count,
            "health_percentage": detection_result.health_percentage,
            "boxes": []  # Could be populated from detection_boxes if needed
        }
        
        # Run analysis
        analysis_result = analytics_service.analyze_detection_result(
            detection_data, 
            detection_result.annotated_image_path,
            session_id=detection_result.session_id
        )
        
        # Save to database
        db_analysis = AnalysisResult(
            analysis_id=analysis_result["analysis_id"],
            detection_result_id=result_id,
            session_id=detection_result.session_id,
            model_used=analysis_result["model_used"],
            processing_time=analysis_result["processing_time"],
            status=analysis_result["status"],
            image_path=detection_result.image_path,
            annotated_image_path=detection_result.annotated_image_path,
            healthy_count=detection_result.healthy_count,
            unhealthy_count=detection_result.unhealthy_count,
            total_count=detection_result.total_count,
            health_percentage=detection_result.health_percentage,
            ai_analysis=analysis_result["ai_analysis"]
        )
        
        db.add(db_analysis)
        db.commit()
        
        logger.info(f"Analysis completed for detection result {result_id}")
        
    except Exception as e:
        logger.error(f"Error in background analysis for result {result_id}: {e}")
        db.rollback()


def analyze_batch_background(session_id: int, batch_analysis_id: str, db: Session):
    """Background task to analyze a batch of detection results."""
    
    try:
        analytics_service = AnalyticsService()
        
        # Get session and results
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            logger.error(f"Session {session_id} not found")
            return
        
        results = db.query(DetectionResult).filter(
            DetectionResult.session_id == session_id,
            DetectionResult.status == "completed"
        ).all()
        
        logger.info(f"Found {len(results)} completed results for session {session_id}")
        
        # Prepare batch data
        batch_data = []
        for result in results:
            batch_data.append({
                "healthy_count": result.healthy_count,
                "unhealthy_count": result.unhealthy_count,
                "total_count": result.total_count,
                "health_percentage": result.health_percentage,
                "annotated_image_path": result.annotated_image_path,
                "image_name": result.image_name
            })
        
        # Run batch analysis (now uses combined image approach)
        logger.info(f"Starting batch analysis for session {session_id}")
        batch_analysis = analytics_service.analyze_batch_results(batch_data, session_id)
        
        # Get the batch analysis record
        batch_record = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.batch_analysis_id == batch_analysis_id
        ).first()
        
        if not batch_record:
            logger.error(f"Batch analysis record {batch_analysis_id} not found")
            return
        
        # Handle analysis results
        if batch_analysis.get("status") in ["single_analysis_only", "batch_combined_analysis"]:
            logger.info(f"Analysis completed for session {session_id}")
            # Save individual analysis to DB if needed
            for analysis in batch_analysis.get("individual_analyses", []):
                # Check if already exists
                existing = db.query(AnalysisResult).filter(
                    AnalysisResult.analysis_id == analysis["analysis_id"]
                ).first()
                if not existing:
                    db_analysis = AnalysisResult(
                        analysis_id=analysis["analysis_id"],
                        detection_result_id=None,
                        session_id=session_id,
                        model_used=analysis["model_used"],
                        processing_time=analysis["processing_time"],
                        status=analysis["status"],
                        image_path=analysis.get("image_path"),
                        annotated_image_path=analysis.get("image_path"),
                        healthy_count=analysis["detection_summary"]["healthy_count"],
                        unhealthy_count=analysis["detection_summary"]["unhealthy_count"],
                        total_count=analysis["detection_summary"]["total_count"],
                        health_percentage=analysis["detection_summary"]["health_percentage"],
                        ai_analysis=analysis["ai_analysis"]
                    )
                    db.add(db_analysis)
                    db.commit()
            
            # Update batch analysis result
            batch_summary = batch_analysis.get("individual_analyses", [{}])[0].get("batch_summary", {})
            if batch_summary:
                batch_record.total_images = batch_summary.get("total_images", len(results))
                batch_record.analyzed_images = batch_summary.get("analyzed_images", len(results))
                batch_record.processing_time = batch_analysis["individual_analyses"][0]["processing_time"]
                batch_record.status = "completed"
                batch_record.total_healthy_leaves = batch_summary.get("total_healthy_leaves", 0)
                batch_record.total_unhealthy_leaves = batch_summary.get("total_unhealthy_leaves", 0)
                batch_record.total_leaves = batch_summary.get("total_leaves", 0)
                batch_record.overall_health_percentage = batch_summary.get("overall_health_percentage", 0.0)
                batch_record.aggregate_recommendations = batch_analysis["individual_analyses"][0]["ai_analysis"]
                batch_record.individual_analysis_ids = [a["analysis_id"] for a in batch_analysis.get("individual_analyses", [])]
                
                db.commit()
                logger.info(f"Successfully updated batch analysis result for session {session_id}")
            
            return
        
        # Handle failed analysis
        logger.error(f"Batch analysis failed for session {session_id}")
        batch_record.status = "failed"
        batch_record.aggregate_recommendations = {"error": "Batch analysis failed"}
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in background batch analysis for session {session_id}: {e}", exc_info=True)
        
        # Update batch record to failed status
        try:
            batch_record = db.query(BatchAnalysisResult).filter(
                BatchAnalysisResult.batch_analysis_id == batch_analysis_id
            ).first()
            if batch_record:
                batch_record.status = "failed"
                batch_record.aggregate_recommendations = {"error": str(e)}
                db.commit()
                logger.info(f"Updated batch analysis record to failed for session {session_id}")
        except Exception as update_error:
            logger.error(f"Failed to update batch analysis record to failed: {update_error}")
            db.rollback()
