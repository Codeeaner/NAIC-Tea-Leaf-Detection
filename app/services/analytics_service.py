"""Analytics service using Qwen3-VL for tea leaf analysis and waste prevention recommendations."""

import os
import base64
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analyzing detection results and providing waste prevention recommendations using Qwen3-VL."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """
        Initialize analytics service.
        
        Args:
            ollama_host: Ollama server host URL
        """
        self.ollama_host = ollama_host
        self.model_name = "qwen3-vl:235b-cloud"
        self.analytics_dir = "analytics"
        
        # Create analytics directory
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Check if Ollama server is available
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama server is available and model is ready."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if not any(self.model_name in name for name in model_names):
                    logger.warning(f"Qwen3-VL model ({self.model_name}) not found. Please install it with: ollama pull {self.model_name}")
                else:
                    logger.info("Ollama server and Qwen3-VL model are available")
            else:
                logger.warning(f"Ollama server not responding correctly: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {self.ollama_host}: {e}")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string for Ollama API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def _create_analysis_prompt(self, detection_data: Dict[str, Any]) -> str:
        """
        Create analysis prompt for Qwen3-VL.
        
        Args:
            detection_data: Detection results data
            
        Returns:
            Formatted prompt string
        """
        healthy_count = detection_data.get("healthy_count", 0)
        unhealthy_count = detection_data.get("unhealthy_count", 0)
        total_count = detection_data.get("total_count", 0)
        health_percentage = detection_data.get("health_percentage", 0.0)
        
        prompt = f"""
You are an expert tea leaf quality analyst and agricultural consultant. I'm showing you an image of tea leaves with detection results overlaid. If the image show no tea leaf, just reply "No tea leaf detected".

DETECTION RESULTS:
- Total leaves detected: {total_count}
- Healthy leaves: {healthy_count}
- Unhealthy leaves: {unhealthy_count}
- Health percentage: {health_percentage:.1f}%

Please analyze this image and provide detailed recommendations in the following JSON format:

{{
    "analysis": {{
        "overall_assessment": "Brief overall assessment of the tea leaf quality based on the image provided",
        "severity_level": "Low/Medium/High",
        "quality_grade": "Premium/Standard/Below standard/Reject"
    }},
    "pollution_diagnosis": {{
        "symptoms_observed": "Choose one symptom from the following list based on the image: Yellowing or bleaching, Black or necrotic spots, Burnt edges or browning tips, Sticky residues / sheen, Deformation/curling, Dusty or dull surface",
        "primary_pollution_source": ["Ozone (O3) exposure", "Sulfur dioxide (SO₂)", "NOx", "Acid rain or excess nitrogen fertilizers", "Pesticide or chemical spray", "Herbicide damage or soil contamination", "Particulate matter (PM10, PM2.5)"],
        "pollution_reduction_solutions": [List the most practical, implementable solutions to reduce exposure to the pollution source, including both immediate protective measures and long-term environmental improvements]
    }},
    "processing_recommendations": {{
        "immediate_actions": ["List of immediate actions to take"],
        "sorting_strategy": "How to sort and separate leaves",
        "processing_method": "Recommended processing approach",
        "quality_preservation": ["Steps to preserve remaining quality"]
    }},
    "waste_prevention": {{
        "salvageable_portions": "What parts can still be used",
        "alternative_uses": ["Alternative uses for defective leaves"],
        "composting_guidelines": "How to compost unusable leaves",
        "prevention_measures": ["How to prevent similar issues in future"]
    }},
    "economic_impact": {{
        "estimated_loss_percentage": "Percentage of economic loss",
        "cost_saving_opportunities": ["Ways to minimize financial impact"],
        "value_recovery_methods": ["Methods to recover some value"]
    }},
    "recommendations": {{
        "priority_actions": ["Top 3 priority actions"],
        "timeline": "Suggested timeline for implementation",
        "monitoring_points": ["What to monitor going forward"]
    }}
}}

For the pollution diagnosis section:
- Only return result if the there is a unhealthy leaf detected or else return empty object
- Examine the tea leaves carefully and identify the most visible symptoms from the provided list
- Based on the symptoms, determine the most likely pollution source
- Provide practical, implementable solutions to reduce exposure to that pollution source
- Include both immediate protective measures and long-term environmental improvements
- Just provide one result for each field, even if multiple symptoms or sources are present

Focus on practical, actionable advice that can help minimize waste and maximize value recovery from the detected tea leaves. Consider both the healthy and unhealthy leaves in your analysis.
"""
        return prompt
    
    def analyze_detection_result(self, detection_data: Dict[str, Any], 
                               annotated_image_path: str, 
                               session_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze detection results using Qwen3-VL and provide recommendations.

        Args:
            detection_data: Detection results from the detection service
            annotated_image_path: Path to the annotated image
            session_id: The ID of the session this analysis belongs to
            
        Returns:
            Analysis results with recommendations
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not annotated_image_path or not os.path.exists(annotated_image_path):
                raise FileNotFoundError(f"Annotated image not found: {annotated_image_path}")
            
            # Encode image to base64
            logger.info(f"Encoding image for analysis: {annotated_image_path}")
            image_b64 = self._encode_image_to_base64(annotated_image_path)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(detection_data)
            
            # Prepare request for Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.4,  # Lower temperature for more consistent analysis
                    "num_predict": 4096   # Allow longer responses
                }
            }
            
            # Send request to Ollama
            logger.info("Sending analysis request to Qwen3-VL...")
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60  # 1 minute timeout for vision model
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            # Parse response
            response_data = response.json()
            llama_response = response_data.get("response", "")
            if not llama_response:
                raise Exception("Empty response from Qwen3-VL")
            logger.info(f"Received response from Qwen3-VL, length: {len(llama_response)} characters")
            analysis_result = self._parse_llama_response(llama_response)
            logger.info(f"Parsed analysis result from response for image: {annotated_image_path}")
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result = {
                "analysis_id": self._generate_analysis_id(),
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "model_used": self.model_name,
                "image_path": annotated_image_path,
                "detection_summary": {
                    "healthy_count": detection_data.get("healthy_count", 0),
                    "unhealthy_count": detection_data.get("unhealthy_count", 0),
                    "total_count": detection_data.get("total_count", 0),
                    "health_percentage": detection_data.get("health_percentage", 0.0)
                },
                "ai_analysis": analysis_result,
                "status": "completed"
            }
            # Save analysis result (formatted JSON)
            self._save_analysis_result(result)
            logger.info(f"Analysis result saved for image: {annotated_image_path}")
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            
            # Return fallback analysis
            return self._create_fallback_analysis(detection_data, annotated_image_path, str(e), session_id)
    
    def _parse_llama_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from Qwen3-VL.
        
        Args:
            response: Raw response string from Qwen3-VL
            
        Returns:
            Parsed analysis data
        """
        try:
            # First, try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        try:
            # Look for JSON block markers (```json...``` or ```...```)
            json_blocks = []
            
            # Find all code blocks
            import re
            code_block_pattern = r'```(?:json)?\s*(.*?)```'
            matches = re.findall(code_block_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    json_blocks.append(parsed)
                except json.JSONDecodeError:
                    continue
            
            # If we found valid JSON blocks, return the first complete one
            if json_blocks:
                return json_blocks[0]
            
            # Fallback: look for JSON-like structure without code blocks
            # Find the first { and try to parse from there
            start_idx = response.find('{')
            if start_idx != -1:
                # Try to find matching closing brace by counting braces
                brace_count = 0
                end_idx = start_idx
                
                for i in range(start_idx, len(response)):
                    char = response[i]
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:  # Found complete JSON
                    json_str = response[start_idx:end_idx]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                
                # If that didn't work, try to the last closing brace
                last_brace = response.rfind('}')
                if last_brace > start_idx:
                    json_str = response[start_idx:last_brace + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # If all parsing attempts failed, use text response fallback
            return self._parse_text_response(response)
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """
        Parse text response when JSON parsing fails.
        
        Args:
            response: Raw text response
            
        Returns:
            Structured analysis data
        """
        # Create basic structure from text response
        return {
            "analysis": {
                "overall_assessment": response[:200] + "..." if len(response) > 200 else response,
                "defect_types": ["Various defects detected"],
                "severity_level": "medium",
                "quality_grade": "standard"
            },
            "processing_recommendations": {
                "immediate_actions": ["Sort leaves by quality", "Separate healthy from unhealthy"],
                "sorting_strategy": "Manual sorting recommended",
                "processing_method": "Standard processing with quality controls",
                "quality_preservation": ["Store in dry conditions", "Process quickly"]
            },
            "waste_prevention": {
                "salvageable_portions": "Healthy portions can be fully utilized",
                "alternative_uses": ["Compost unhealthy leaves", "Use for fertilizer"],
                "composting_guidelines": "Standard composting procedures",
                "prevention_measures": ["Regular quality monitoring", "Improved harvesting practices"]
            },
            "economic_impact": {
                "estimated_loss_percentage": "15-25%",
                "cost_saving_opportunities": ["Improved sorting", "Better processing"],
                "value_recovery_methods": ["Alternative products", "Composting"]
            },
            "recommendations": {
                "priority_actions": ["Sort immediately", "Process healthy leaves first", "Plan waste utilization"],
                "timeline": "Immediate action recommended",
                "monitoring_points": ["Quality trends", "Processing efficiency"]
            },
            "raw_response": response
        }
    
    def _create_fallback_analysis(self, detection_data: Dict[str, Any], 
                                image_path: str, error: str, session_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create fallback analysis when AI analysis fails.
        
        Args:
            detection_data: Detection results
            image_path: Path to image
            error: Error message
            session_id: The ID of the session this analysis belongs to
            
        Returns:
            Fallback analysis result
        """
        healthy_count = detection_data.get("healthy_count", 0)
        unhealthy_count = detection_data.get("unhealthy_count", 0)
        total_count = detection_data.get("total_count", 0)
        health_percentage = detection_data.get("health_percentage", 0.0)
        
        # Determine severity based on health percentage
        if health_percentage >= 80:
            severity = "low"
            grade = "premium"
        elif health_percentage >= 60:
            severity = "medium"
            grade = "standard"
        else:
            severity = "high"
            grade = "below_standard"
        
        return {
            "analysis_id": self._generate_analysis_id(),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0.0,
            "model_used": "fallback_analysis",
            "image_path": image_path,
            "detection_summary": {
                "healthy_count": healthy_count,
                "unhealthy_count": unhealthy_count,
                "total_count": total_count,
                "health_percentage": health_percentage
            },
            "ai_analysis": {
                "analysis": {
                    "overall_assessment": f"Automated analysis based on detection results. {health_percentage:.1f}% healthy leaves detected.",
                    "defect_types": ["Quality assessment based on detection"],
                    "severity_level": severity,
                    "quality_grade": grade
                },
                "processing_recommendations": {
                    "immediate_actions": [
                        "Sort leaves by quality immediately",
                        "Separate healthy from unhealthy leaves",
                        "Process healthy leaves first"
                    ],
                    "sorting_strategy": "Automated sorting recommended based on detection results",
                    "processing_method": "Standard processing with quality controls",
                    "quality_preservation": [
                        "Maintain proper storage conditions",
                        "Process within optimal timeframe"
                    ]
                },
                "waste_prevention": {
                    "salvageable_portions": f"{healthy_count} healthy leaves can be fully utilized",
                    "alternative_uses": [
                        "Compost unhealthy leaves for organic fertilizer",
                        "Use lower grade leaves for secondary products"
                    ],
                    "composting_guidelines": "Standard composting procedures for organic waste",
                    "prevention_measures": [
                        "Regular quality monitoring",
                        "Improved harvesting practices",
                        "Better storage conditions"
                    ]
                },
                "economic_impact": {
                    "estimated_loss_percentage": f"{100 - health_percentage:.1f}%",
                    "cost_saving_opportunities": [
                        "Improved sorting efficiency",
                        "Better processing methods"
                    ],
                    "value_recovery_methods": [
                        "Alternative product development",
                        "Organic fertilizer production"
                    ]
                },
                "recommendations": {
                    "priority_actions": [
                        "Sort leaves immediately",
                        "Process healthy leaves first",
                        "Plan waste utilization strategy"
                    ],
                    "timeline": "Immediate action recommended for fresh leaves",
                    "monitoring_points": [
                        "Quality trends over time",
                        "Processing efficiency metrics"
                    ]
                }
            },
            "status": "completed_fallback",
            "error": error
        }
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        import uuid
        timestamp = int(datetime.now().timestamp())
        unique_id = uuid.uuid4().hex[:8]
        return f"analysis_{unique_id}_{timestamp}"
    
    def _save_analysis_result(self, analysis_result: Dict[str, Any]):
        """
        Save analysis result to file.
        
        Args:
            analysis_result: Analysis result to save
        """
        try:
            analysis_id = analysis_result.get("analysis_id")
            
            # Don't save if analysis_id is missing or contains "unknown"
            if not analysis_id or "unknown" in analysis_id:
                logger.warning(f"Skipping save for invalid analysis_id: {analysis_id}")
                return
            
            filename = f"{analysis_id}.json"
            filepath = os.path.join(self.analytics_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis result saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
    
    def _combine_images(self, image_paths: List[str]) -> str:
        """
        Combine multiple images into a single grid image.
        
        Args:
            image_paths: List of paths to images to combine
            
        Returns:
            Path to the combined image file
        """
        try:
            from PIL import Image
            import math
            
            # Load all images
            images = []
            for path in image_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    images.append(img)
            
            if not images:
                raise ValueError("No valid images to combine")
            
            # Calculate grid dimensions (try to make it roughly square)
            num_images = len(images)
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)
            
            # Resize all images to same size (use the size of first image)
            target_size = images[0].size
            resized_images = []
            for img in images:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
            
            # Create combined image
            combined_width = target_size[0] * cols
            combined_height = target_size[1] * rows
            combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
            
            # Paste images into grid
            for i, img in enumerate(resized_images):
                row = i // cols
                col = i % cols
                x = col * target_size[0]
                y = row * target_size[1]
                combined_image.paste(img, (x, y))
            
            # Save combined image
            timestamp = int(datetime.now().timestamp())
            combined_filename = f"combined_batch_{timestamp}.jpg"
            combined_path = os.path.join("results", combined_filename)
            combined_image.save(combined_path, "JPEG", quality=90)
            
            logger.info(f"Combined {len(images)} images into: {combined_path}")
            return combined_path
            
        except Exception as e:
            logger.error(f"Error combining images: {e}")
            raise

    def analyze_batch_results(self, batch_results: List[Dict[str, Any]], session_id: int) -> Dict[str, Any]:
        """
        Analyze a batch of detection results by combining all images and using single analysis.
        """
        start_time = datetime.now()
        
        try:
            total_images = len(batch_results)
            total_healthy = sum(r.get("healthy_count", 0) for r in batch_results)
            total_unhealthy = sum(r.get("unhealthy_count", 0) for r in batch_results)
            total_leaves = total_healthy + total_unhealthy
            overall_health_percentage = (total_healthy / total_leaves * 100) if total_leaves > 0 else 0

            # If only one result, use individual analysis
            if total_images == 1:
                single_result = batch_results[0]
                analysis = self.analyze_detection_result(
                    single_result,
                    single_result["annotated_image_path"],
                    session_id=session_id
                )
                return {
                    "individual_analyses": [analysis],
                    "status": "single_analysis_only"
                }

            # For multiple images: combine them into one image and analyze
            image_paths = [r["annotated_image_path"] for r in batch_results if r.get("annotated_image_path") and os.path.exists(r["annotated_image_path"])]
            
            if not image_paths:
                raise Exception("No valid annotated images found for batch analysis")
            
            # Combine all images into one
            combined_image_path = self._combine_images(image_paths)
            
            # Create combined detection data
            combined_detection_data = {
                "healthy_count": total_healthy,
                "unhealthy_count": total_unhealthy,
                "total_count": total_leaves,
                "health_percentage": overall_health_percentage
            }
            
            # Use single analysis on the combined image
            logger.info(f"Analyzing combined image with {total_images} sub-images")
            analysis = self.analyze_detection_result(
                combined_detection_data,
                combined_image_path,
                session_id=session_id
            )
            
            # Update analysis to indicate it's a batch analysis
            analysis["analysis_type"] = "batch_combined"
            analysis["batch_summary"] = {
                "total_images": total_images,
                "analyzed_images": total_images,
                "total_healthy_leaves": total_healthy,
                "total_unhealthy_leaves": total_unhealthy,
                "total_leaves": total_leaves,
                "overall_health_percentage": overall_health_percentage,
                "combined_image_path": combined_image_path
            }
            
            return {
                "individual_analyses": [analysis],
                "status": "batch_combined_analysis"
            }

        except Exception as e:
            logger.error(f"Error during batch analysis: {e}")
            failed_analysis_id = self._generate_analysis_id()
            return {
                "batch_analysis_id": failed_analysis_id,
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "error": str(e)
            }

    def get_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get history of analysis results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        try:
            analysis_files = []
            
            if os.path.exists(self.analytics_dir):
                for filename in os.listdir(self.analytics_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.analytics_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                data['filename'] = filename
                                analysis_files.append(data)
                        except Exception as e:
                            logger.warning(f"Failed to read analysis file {filename}: {e}")
            
            # Sort by timestamp and limit results
            analysis_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return analysis_files[:limit]
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []