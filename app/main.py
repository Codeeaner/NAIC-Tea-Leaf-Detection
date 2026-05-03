"""Main FastAPI application for tea leaf detection website."""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import API routers
from app.api.upload import router as upload_router
from app.api.detection import router as detection_router
from app.api.reports import router as reports_router
from app.api.analytics import router as analytics_router

# Import database setup
from app.database import create_tables, engine
from app.models.detection import Base
from app.models.analytics import *  # Import analytics models for table creation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Tea Leaf Detection API...")
    
    # Create database tables
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
    
    # Create required directories
    directories = ["uploads", "results", "reports", "analytics", "static", "static/images"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Tea Leaf Detection API...")


# Create FastAPI application
app = FastAPI(
    title="Tea Leaf Detection API",
    description="AI-powered tea leaf health detection and analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API routers
app.include_router(upload_router)
app.include_router(detection_router)
app.include_router(reports_router)
app.include_router(analytics_router, prefix="/api/analytics")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "huggingface_space_url": os.getenv("HUGGINGFACE_SPACE_URL"),
        },
    )


@app.get("/results-page", response_class=HTMLResponse)
async def results_page(request: Request):
    """Serve the results page."""
    return templates.TemplateResponse("results.html", {"request": request})


@app.get("/history-page", response_class=HTMLResponse)
async def history_page(request: Request):
    """Serve the history page."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the farmer-friendly analytics dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: int):
    """Serve the session detail page with AI analysis."""
    try:
        from httpx import AsyncClient
        async with AsyncClient(base_url="http://localhost:8000") as client:
            # Fetch session details
            session_response = await client.get(f"/api/results/{session_id}")
            if session_response.status_code != 200:
                raise ValueError(f"Failed to fetch session data: {session_response.status_code} {session_response.text}")
            session_data = session_response.json()
            
            # Calculate statistics
            completed_results = [r for r in session_data.get('results', []) if r.get('status') == 'completed']
            failed_results = [r for r in session_data.get('results', []) if r.get('status') == 'failed']
            completed_count = len(completed_results)
            failed_count = len(failed_results)
            total_healthy = sum(r.get('healthy_count', 0) for r in completed_results)
            total_unhealthy = sum(r.get('unhealthy_count', 0) for r in completed_results)
            total_leaves = total_healthy + total_unhealthy
            average_health = (total_healthy / total_leaves * 100) if total_leaves > 0 else 0
            
            # Fetch AI analysis if available
            analysis_data = None
            try:
                analysis_response = await client.get(f"/api/analytics/batch/{session_id}")
                if analysis_response.status_code == 200:
                    analysis_data = analysis_response.json()
                else:
                    logger.warning(f"Analysis data not found or error: {analysis_response.status_code}")
            except Exception as ae:
                logger.error(f"Error fetching analysis data: {ae}")
            
            return templates.TemplateResponse("session_detail.html", {
                "request": request,
                "session": session_data,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "total_healthy": total_healthy,
                "total_unhealthy": total_unhealthy,
                "total_leaves": total_leaves,
                "average_health": average_health,
                "analysis": analysis_data
            })
    except Exception as e:
        logger.error(f"Error loading session details for ID {session_id}: {str(e)}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "error": f"Failed to load session details: {str(e)}", "status_code": 500},
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Tea Leaf Detection API",
        "version": "1.0.0"
    }


@app.get("/api/info")
async def get_api_info():
    """Get API information and model status."""
    try:
        from app.services.detection_service import TeaLeafDetectionService
        
        detection_service = TeaLeafDetectionService()
        model_info = detection_service.get_model_info()
        
        return {
            "api_name": "Tea Leaf Detection API",
            "version": "1.0.0",
            "description": "AI-powered tea leaf health detection and analysis",
            "features": [
                "Single image detection",
                "Batch image processing",
                "Detection history",
                "PDF and CSV reports",
                "Real-time progress tracking",
                "AI-powered analytics with Qwen3-VL",
                "Farmer decision support for sell/process/discard routing",
                "Waste prevention recommendations",
                "Quality metrics tracking"
            ],
            "model_info": model_info,
            "endpoints": {
                "upload_single": "/api/upload/single",
                "upload_batch": "/api/upload/batch",
                "get_results": "/api/results/{session_id}",
                "get_history": "/api/history",
                "get_stats": "/api/stats",
                "download_pdf": "/api/reports/download/pdf/{session_id}",
                "download_csv": "/api/reports/download/csv/{session_id}",
                "analyze_result": "/api/analytics/analyze/{result_id}",
                "analyze_batch": "/api/analytics/analyze/batch/{session_id}",
                "get_analysis": "/api/analytics/results/{analysis_id}",
                "get_batch_analysis": "/api/analytics/batch/{batch_analysis_id}",
                "get_recommendations": "/api/analytics/recommendations/{analysis_id}",
                "decision_support": "/api/analytics/decision-support/{session_id}",
                "analytics_history": "/api/analytics/history"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting API info: {e}")
        return {
            "api_name": "Tea Leaf Detection API",
            "version": "1.0.0",
            "status": "error",
            "error": "Failed to load model information"
        }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return templates.TemplateResponse(
        "error.html", 
        {"request": request, "error": "Page not found", "status_code": 404},
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "error": "Internal server error", "status_code": 500},
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
