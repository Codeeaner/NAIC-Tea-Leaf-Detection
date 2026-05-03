"""
Professional Tea Leaf Detection System
Real-time detection using YOLOv8 model with camera input
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TeaLeafDetector:
    """Professional tea leaf detection system using AI model"""

    def __init__(self, model_path=r'C:\Users\amber\2025\T4G\Tea Leaf\runs\detect\train\weights\best.pt', confidence_threshold=0.5):
        """
        Initialize the tea leaf detector
        
        Args:
            model_path (str): Path to the AI model file
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        
        # Colors for different classes (BGR format)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load the AI model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading AI model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _initialize_camera(self, camera_index=0):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_index}")
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {camera_index} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def _draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            
        Returns:
            Annotated frame with bounding boxes
        """
        annotated_frame = frame.copy()
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Skip low confidence detections
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get class name
                    class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"Class_{class_id}"
                    
                    # Select color based on class
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label text
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate label size and position
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return annotated_frame
    
    def _add_info_overlay(self, frame, fps, detection_count):
        """Add information overlay to the frame"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add text information
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run_detection(self, camera_index=0, save_output=False, output_path="output.avi"):
        """
        Run real-time tea leaf detection
        
        Args:
            camera_index (int): Camera index to use
            save_output (bool): Whether to save output video
            output_path (str): Path to save output video
        """
        try:
            self._initialize_camera(camera_index)
            
            # Initialize video writer if saving output
            video_writer = None
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Saving output to: {output_path}")
            
            # FPS calculation variables
            prev_time = time.time()
            fps = 0
            
            logger.info("Starting tea leaf detection. Press 'q' to quit.")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Run inference
                results = self.model(frame, verbose=False)
                
                # Count detections
                detection_count = 0
                if results and len(results) > 0 and results[0].boxes is not None:
                    detection_count = len([box for box in results[0].boxes 
                                         if box.conf[0] >= self.confidence_threshold])
                
                # Draw detections
                annotated_frame = self._draw_detections(frame, results)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Add information overlay
                final_frame = self._add_info_overlay(annotated_frame, fps, detection_count)
                
                # Save frame if recording
                if video_writer is not None:
                    video_writer.write(final_frame)
                
                # Display frame
                cv2.imshow('Tea Leaf Detection', final_frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"tea_leaf_detection_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, final_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            self._cleanup(video_writer)
    
    def _cleanup(self, video_writer=None):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")
    
    def detect_image(self, image_path, output_path=None):
        """
        Detect tea leaves in a single image
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Run inference
            results = self.model(image, verbose=False)
            
            # Draw detections
            annotated_image = self._draw_detections(image, results)
            
            # Count detections
            detection_count = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                detection_count = len([box for box in results[0].boxes 
                                     if box.conf[0] >= self.confidence_threshold])
            
            logger.info(f"Detected {detection_count} tea leaves in {image_path}")
            
            # Save or display result
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Annotated image saved: {output_path}")
            else:
                cv2.imshow('Tea Leaf Detection - Image', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return annotated_image, detection_count
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise


def main():
    """Main function to run the tea leaf detector"""
    try:
        # Initialize detector
        detector = TeaLeafDetector(
            model_path=r'C:\Users\amber\2025\T4G\Tea Leaf\runs\detect\train\weights\best.pt',
            confidence_threshold=0.2
        )
        
        print("Tea Leaf Detection System")
        print("=" * 30)
        print("1. Real-time camera detection")
        print("2. Process single image")
        print("3. Real-time with video recording")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Real-time detection
            detector.run_detection(camera_index=0)
            
        elif choice == "2":
            # Single image processing
            image_path = input("Enter image path: ").strip()
            output_path = input("Enter output path (optional): ").strip()
            if not output_path:
                output_path = None
            
            detector.detect_image(image_path, output_path)
            
        elif choice == "3":
            # Real-time with recording
            output_path = input("Enter output video path (default: tea_detection.avi): ").strip()
            if not output_path:
                output_path = "tea_detection.avi"
            
            detector.run_detection(camera_index=0, save_output=True, output_path=output_path)
        
        else:
            print("Invalid choice")
    
    except Exception as e:
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
