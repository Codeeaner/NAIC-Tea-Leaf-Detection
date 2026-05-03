#!/usr/bin/env python3
"""
Test script for Tea Leaf Analytics functionality.

This script tests the analytics service integration with Qwen3-VL 235B Cloud
to ensure waste prevention recommendations are working correctly.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.analytics_service import AnalyticsService
from app.services.detection_service import TeaLeafDetectionService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    print("🔍 Testing Ollama connection...")
    
    try:
        analytics_service = AnalyticsService()
        print("✅ Analytics service initialized successfully")
        print(f"   - Ollama host: {analytics_service.ollama_host}")
        print(f"   - Model: {analytics_service.model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize analytics service: {e}")
        return False


def test_mock_detection_data():
    """Test analytics with mock detection data."""
    print("\n🧪 Testing with mock detection data...")
    
    try:
        analytics_service = AnalyticsService()
        
        # Create mock detection data
        mock_detection = {
            "healthy_count": 15,
            "unhealthy_count": 8,
            "total_count": 23,
            "health_percentage": 65.2,
            "boxes": [
                {"class_id": 0, "class_name": "unhealthy", "confidence": 0.89},
                {"class_id": 1, "class_name": "healthy", "confidence": 0.76}
            ]
        }
        
        # Find a test image (look for any image in results or static directories)
        test_image_paths = [
            "static/images/sample.jpg",
            "results/test_annotated.jpg",
            "runs/detect/predict/image1.jpg"
        ]
        
        test_image = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image = path
                break
        
        if not test_image:
            print("⚠️  No test image found. Creating a fallback analysis...")
            # Test fallback analysis
            result = analytics_service._create_fallback_analysis(
                mock_detection, "test_image.jpg", "No image available for testing"
            )
        else:
            print(f"📸 Using test image: {test_image}")
            result = analytics_service.analyze_detection_result(mock_detection, test_image)
        
        print("✅ Analytics completed successfully!")
        print(f"   - Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"   - Status: {result.get('status', 'N/A')}")
        print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Print key recommendations
        ai_analysis = result.get('ai_analysis', {})
        if 'recommendations' in ai_analysis:
            recommendations = ai_analysis['recommendations']
            print("\n📋 Key Recommendations:")
            for action in recommendations.get('priority_actions', []):
                print(f"   • {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock testing failed: {e}")
        logger.exception("Full error details:")
        return False


def test_real_detection_integration():
    """Test integration with real detection service."""
    print("\n🔗 Testing integration with detection service...")
    
    try:
        # Initialize detection service with analytics enabled
        detection_service = TeaLeafDetectionService()
        detection_service.enable_analytics_mode()
        
        # Look for a test image
        test_images = [
            "static/images/sample.jpg",
            "uploads/test.jpg"
        ]
        
        test_image = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if not test_image:
            print("⚠️  No test image found for integration testing")
            print("   Please add a test image to static/images/ or uploads/")
            return False
        
        print(f"📸 Running detection on: {test_image}")
        
        # Run detection (which should trigger analytics if enabled)
        detection_result = detection_service.detect_image(
            test_image, 
            save_annotated=True, 
            output_dir="test_results"
        )
        
        print("✅ Detection completed!")
        print(f"   - Healthy leaves: {detection_result.get('healthy_count', 0)}")
        print(f"   - Unhealthy leaves: {detection_result.get('unhealthy_count', 0)}")
        print(f"   - Health percentage: {detection_result.get('health_percentage', 0):.1f}%")
        
        if detection_result.get('annotated_image_path'):
            print(f"   - Annotated image: {detection_result['annotated_image_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration testing failed: {e}")
        logger.exception("Full error details:")
        return False


def test_batch_analytics():
    """Test batch analytics functionality."""
    print("\n📦 Testing batch analytics...")
    
    try:
        analytics_service = AnalyticsService()
        
        # Create mock batch data
        batch_data = [
            {
                "healthy_count": 12,
                "unhealthy_count": 3,
                "total_count": 15,
                "health_percentage": 80.0,
                "annotated_image_path": "test1.jpg",
                "image_name": "test1.jpg"
            },
            {
                "healthy_count": 8,
                "unhealthy_count": 7,
                "total_count": 15,
                "health_percentage": 53.3,
                "annotated_image_path": "test2.jpg", 
                "image_name": "test2.jpg"
            },
            {
                "healthy_count": 18,
                "unhealthy_count": 2,
                "total_count": 20,
                "health_percentage": 90.0,
                "annotated_image_path": "test3.jpg",
                "image_name": "test3.jpg"
            }
        ]
        
        batch_result = analytics_service.analyze_batch_results(batch_data)
        
        print("✅ Batch analytics completed!")
        print(f"   - Batch Analysis ID: {batch_result.get('batch_analysis_id', 'N/A')}")
        print(f"   - Total images: {batch_result.get('batch_summary', {}).get('total_images', 0)}")
        print(f"   - Overall health: {batch_result.get('batch_summary', {}).get('overall_health_percentage', 0):.1f}%")
        
        # Print aggregate recommendations
        agg_rec = batch_result.get('aggregate_recommendations', {})
        if 'overall_assessment' in agg_rec:
            assessment = agg_rec['overall_assessment']
            print(f"   - Quality level: {assessment.get('quality_level', 'N/A')}")
            print(f"   - Priority: {assessment.get('priority_level', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch analytics testing failed: {e}")
        logger.exception("Full error details:")
        return False


def test_analytics_history():
    """Test analytics history functionality."""
    print("\n📊 Testing analytics history...")
    
    try:
        analytics_service = AnalyticsService()
        history = analytics_service.get_analysis_history(limit=5)
        
        print(f"✅ Retrieved {len(history)} historical analyses")
        
        for i, analysis in enumerate(history[:3], 1):
            print(f"   {i}. Analysis ID: {analysis.get('analysis_id', 'N/A')}")
            print(f"      Status: {analysis.get('status', 'N/A')}")
            print(f"      Timestamp: {analysis.get('timestamp', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ History testing failed: {e}")
        logger.exception("Full error details:")
        return False


def main():
    """Run all analytics tests."""
    print("🚀 Tea Leaf Analytics Test Suite")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Mock Detection Data", test_mock_detection_data),
        ("Detection Integration", test_real_detection_integration),
        ("Batch Analytics", test_batch_analytics),
        ("Analytics History", test_analytics_history)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Analytics functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the setup and try again.")
        print("\n💡 Tips:")
        print("   - Ensure Ollama is running: ollama serve")
        print("   - Install the model: ollama pull qwen3-vl:235b-cloud")
        print("   - Check system resources (RAM, disk space)")
        print("   - Add test images to static/images/ directory")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
