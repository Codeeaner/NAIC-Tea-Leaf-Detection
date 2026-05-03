import os
import json
import time
import logging
import base64
from pathlib import Path
from typing import Dict, Any

import requests
import cv2
import numpy as np
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://acwz-tealeafdetection.hf.space"


def extract_image_url(data: Any) -> str:
    """Extract annotated image URL from Gradio response recursively."""
    if isinstance(data, str):
        if "gradio_api/file" in data:
            return data
        return None

    if isinstance(data, list):
        for item in data:
            result = extract_image_url(item)
            if result:
                return result

    if isinstance(data, dict):
        for v in data.values():
            result = extract_image_url(v)
            if result:
                return result

    return None


def run_detection(image_path: str, confidence: float):
    with open(image_path, "rb") as file_handle:
        upload_response = requests.post(
            f"{BASE_URL}/gradio_api/upload",
            files={"files": file_handle},
            timeout=120,
        )
    upload_response.raise_for_status()

    file_path = upload_response.json()[0]

    prediction_response = requests.post(
        f"{BASE_URL}/gradio_api/call/v2/predict",
        json={
            "image": {
                "path": file_path,
                "meta": {"_type": "gradio.FileData"},
            },
            "confidence": confidence,
        },
        timeout=120,
    )
    prediction_response.raise_for_status()

    event_id = prediction_response.json()["event_id"]

    stream_response = requests.get(
        f"{BASE_URL}/gradio_api/call/predict/{event_id}",
        stream=True,
        timeout=120,
    )
    stream_response.raise_for_status()

    final_payload = None
    event_type = None

    for line in stream_response.iter_lines(decode_unicode=True):
        if not line:
            continue

        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()

        if line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            try:
                final_payload = json.loads(data)
            except json.JSONDecodeError:
                final_payload = data

            if event_type == "complete":
                break

    # IMPORTANT: Gradio returns [image, json]
    return final_payload


class TeaLeafDetectionService:
    """Service for tea leaf detection using YOLO model."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Initialize the tea leaf detection service.

        Args:
            model_path: Unused. Kept for backward compatibility.
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = None
        self.confidence_threshold = confidence_threshold

        self.class_names = {0: "unhealthy", 1: "healthy"}
        self.colors = {
            0: (0, 0, 255),
            1: (0, 255, 0),
        }

    def detect_image(self, image_path: str, save_annotated: bool = True,
                    output_dir: str = "results") -> Dict[str, Any]:
        """
        Detect tea leaves in a single image.

        Args:
            image_path: Path to the input image
            save_annotated: Whether to save annotated image
            output_dir: Directory to save results

        Returns:
            Dictionary with detection results
        """
        start_time = time.time()

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            logger.info("Running remote detection via Hugging Face Space API: acwz-tealeafdetection")
            remote_result = self._detect_image_via_huggingface_space(
                image_path=image_path,
                image=image,
                save_annotated=save_annotated,
                output_dir=output_dir,
                start_time=start_time,
            )
            return remote_result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "error": str(e),
                "healthy_count": 0,
                "unhealthy_count": 0,
                "total_count": 0,
                "health_percentage": 0.0,
                "boxes": [],
                "processing_time": time.time() - start_time,
                "annotated_image": None,
            }

    def _detect_image_via_huggingface_space(self, image_path: str, image: np.ndarray,
                                            save_annotated: bool, output_dir: str,
                                            start_time: float) -> Dict[str, Any]:
        """Send the image through the Gradio API flow from test.py and normalize the result."""
        payload = run_detection(image_path, self.confidence_threshold)
        detections = self._extract_remote_detections(payload)
        boxes_data = self._normalize_remote_boxes(detections)

        healthy_count = 0
        unhealthy_count = 0
        for box in boxes_data:
            class_id = box.get("class_id")
            class_name = str(box.get("class_name", "")).lower()

            if class_id == 0 or class_name == "unhealthy":
                unhealthy_count += 1
            elif class_id == 1 or class_name == "healthy":
                healthy_count += 1

        total_count = healthy_count + unhealthy_count
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0.0

        annotated_image_data = None
        annotated_image_path = None
        if save_annotated:
            # Extract the annotated image URL from the remote response
            annotated_image_url = extract_image_url(payload)
            if annotated_image_url:
                try:
                    # Fetch the annotated image
                    img_response = requests.get(annotated_image_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Save to disk
                    os.makedirs(output_dir, exist_ok=True)
                    original_name = Path(image_path).stem
                    unique_id = str(uuid.uuid4())[:8]
                    output_filename = f"{original_name}_{unique_id}_annotated.jpg"
                    annotated_image_path = os.path.join(output_dir, output_filename)
                    
                    with open(annotated_image_path, "wb") as f:
                        f.write(img_response.content)
                    
                    # Encode as base64 for API response
                    annotated_image_data = base64.b64encode(img_response.content).decode('utf-8')
                    logger.info(f"Successfully fetched and saved annotated image from Space: {annotated_image_path}")
                except Exception as e:
                    logger.warning(f"Failed to fetch annotated image from Space: {e}")
            else:
                logger.warning("No annotated image URL found in Space response")

        logger.info(
            "Remote detection summary - Healthy: %s, Unhealthy: %s, Total: %s",
            healthy_count,
            unhealthy_count,
            total_count,
        )

        return {
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "total_count": total_count,
            "health_percentage": health_percentage,
            "boxes": boxes_data,
            "annotated_image_path": annotated_image_path,
            "annotated_image": annotated_image_data,
            "processing_time": time.time() - start_time,
            "source": "huggingface_space_api",
        }

    def _extract_remote_detections(self, payload: Any) -> Any:
        """Extract the detections payload from a Gradio response."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return payload

        if isinstance(payload, list):
            return payload[-1] if payload else []

        if isinstance(payload, dict):
            for key in ("data", "result", "output", "outputs", "predictions", "detections"):
                value = payload.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    return value[-1] if value else []
                if isinstance(value, dict):
                    nested = self._extract_remote_detections(value)
                    if nested is not value:
                        return nested
                    return value
                if isinstance(value, str):
                    try:
                        return self._extract_remote_detections(value)
                    except Exception:
                        return value

            return payload

        return payload

    def _normalize_remote_boxes(self, detections: Any) -> list:
        """Convert remote detections into the local box format used by the UI."""
        if isinstance(detections, str):
            try:
                detections = json.loads(detections)
            except json.JSONDecodeError:
                return []

        if isinstance(detections, dict):
            for key in ("detections", "predictions", "data", "items"):
                value = detections.get(key)
                if isinstance(value, list):
                    detections = value
                    break
            else:
                detections = [detections]

        if not isinstance(detections, list):
            return []

        boxes_data = []
        for detection in detections:
            if isinstance(detection, str):
                try:
                    detection = json.loads(detection)
                except json.JSONDecodeError:
                    continue

            if not isinstance(detection, dict):
                continue

            box = detection.get("box") if isinstance(detection.get("box"), dict) else {}
            class_id = detection.get("class")
            if class_id is None:
                class_id = detection.get("class_id")
            if isinstance(class_id, str) and class_id.isdigit():
                class_id = int(class_id)

            class_name = detection.get("name") or detection.get("class_name") or detection.get("label")
            confidence = detection.get("confidence")
            if confidence is None:
                confidence = detection.get("score")

            x1 = detection.get("x1", box.get("x1", box.get("xmin", box.get("left"))))
            y1 = detection.get("y1", box.get("y1", box.get("ymin", box.get("top"))))
            x2 = detection.get("x2", box.get("x2", box.get("xmax")))
            y2 = detection.get("y2", box.get("y2", box.get("ymax")))

            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue

            boxes_data.append({
                "class_id": int(class_id) if class_id is not None else None,
                "class_name": class_name or (self.class_names.get(int(class_id)) if class_id is not None else "unknown"),
                "confidence": float(confidence) if confidence is not None else 0.0,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

        return boxes_data


    def _create_annotated_image(self, image: np.ndarray, boxes_data: list,
                               original_image_path: str, output_dir: str) -> str:
        """Create and save annotated image with detection boxes."""
        os.makedirs(output_dir, exist_ok=True)

        annotated_image = image.copy()

        for box_data in boxes_data:
            x1, y1, x2, y2 = int(box_data["x1"]), int(box_data["y1"]), int(box_data["x2"]), int(box_data["y2"])
            class_id = box_data["class_id"]
            class_name = box_data["class_name"]
            confidence = box_data["confidence"]

            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        original_name = Path(original_image_path).stem
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{original_name}_{unique_id}_annotated.jpg"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, annotated_image)
        return output_path

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": None,
            "model_loaded": False,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.class_names,
            "model_type": "Hugging Face Space API",
            "remote_space_base_url": BASE_URL,
        }
