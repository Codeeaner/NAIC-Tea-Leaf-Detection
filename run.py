#!/usr/bin/env python3
"""
Tea Leaf Detection Website Startup Script

This script provides an easy way to start the tea leaf detection web application
with proper initialization and configuration.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'pydantic', 
        'pillow', 'opencv-python', 'ultralytics', 'pandas', 
        'plotly', 'reportlab', 'python-multipart', 'jinja2', 'aiofiles'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_model():
    """Check if the YOLO model exists."""
    model_paths = [
        "runs/detect/train/weights/best.pt",
        "best.pt",
        os.path.join(os.path.dirname(__file__), "runs", "detect", "train", "weights", "best.pt")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found YOLO model at: {path}")
            return path
    
    print("❌ YOLO model not found. Expected locations:")
    for path in model_paths:
        print(f"   - {path}")
    print("\n🔧 Please ensure your trained model file is available")
    return None

def create_directories():
    """Create required directories."""
    directories = [
        "uploads", "results", "reports", "static", 
        "static/css", "static/js", "static/images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def initialize_database():
    """Initialize the database with required tables."""
    try:
        from app.database import create_tables
        create_tables()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main function to start the application."""
    parser = argparse.ArgumentParser(description="Tea Leaf Detection Website")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and model checks")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    print("🍃 Tea Leaf Detection Website")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    
    if not args.skip_checks:
        print("\n🔍 Running pre-flight checks...")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check model
        model_path = check_model()
        if not model_path:
            response = input("\n❓ Continue without model check? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Create directories
        print("\n📁 Setting up directories...")
        create_directories()
        
        # Initialize database
        print("\n💾 Initializing database...")
        if not initialize_database():
            response = input("\n❓ Continue despite database error? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    print("\n🚀 Starting Tea Leaf Detection Website...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Reload: {args.reload}")
    print(f"   Log Level: {args.log_level}")
    
    print(f"\n🌐 Application will be available at:")
    print(f"   Web Interface: http://{args.host}:{args.port}")
    print(f"   API Documentation: http://{args.host}:{args.port}/docs")
    
    print(f"\n📊 Features available:")
    print(f"   ✅ Single image detection")
    print(f"   ✅ Batch image processing")
    print(f"   ✅ Detection history")
    print(f"   ✅ PDF and CSV reports")
    print(f"   ✅ RESTful API")
    
    # Start the application
    try:
        import uvicorn
        
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Tea Leaf Detection Website...")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()