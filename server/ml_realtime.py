"""
Real-Time Sketch Feedback System
Provides live feedback as users draw sketches with low-latency inference.
"""

import asyncio
import base64
import numpy as np
import cv2
import torch
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import threading

class RealtimeFeedbackEngine:
    """Handles real-time sketch analysis and feedback generation."""
    
    def __init__(self, model_manager, faiss_indexer, db_manager, cache_size=10):
        self.model_manager = model_manager
        self.faiss_indexer = faiss_indexer
        self.db_manager = db_manager
        
        # Cache for recent embeddings to avoid redundant computation
        self.embedding_cache = deque(maxlen=cache_size)
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)
        self.last_feedback_time = 0
        self.feedback_debounce_ms = 300  # Minimum time between feedback updates
    
    def preprocess_sketch(self, image_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess sketch image for analysis.
        Returns: (preprocessed_image, original_image)
        """
        try:
            # Decode image
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Store original for visualization
            original = image_array.copy()
            
            # Resize to model input size
            image_array = cv2.resize(image_array, (224, 224))
            
            # Normalize
            image_array = image_array.astype(np.float32) / 255.0
            
            return image_array, original
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, None
    
    def extract_features(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D embedding from sketch."""
        try:
            start_time = time.time()
            
            # Convert to tensor and match training-time channel layout (1, 3, 224, 224)
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.repeat(1, 3, 1, 1).float()
            
            # Get embedding from model
            embedding = self.model_manager.get_embedding(image_tensor)
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return embedding
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def analyze_sketch_quality(self, image_array: np.ndarray) -> Dict:
        """
        Analyze sketch quality and provide feedback.
        Returns: feedback dictionary with issues and suggestions
        """
        feedback = {
            "symmetry_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "issues": [],
            "suggestions": [],
            "overall_quality": 0.0
        }
        
        try:
            # Edge detection for feature analysis
            edges = cv2.Canny((image_array * 255).astype(np.uint8), 50, 150)
            
            # Detect contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                feedback["issues"].append("No facial features detected")
                feedback["suggestions"].append("Start drawing facial features")
                return feedback
            
            # Analyze symmetry (left vs right halves)
            h, w = image_array.shape
            left_half = image_array[:, :w//2]
            right_half = image_array[:, w//2:]
            
            # Flip right half for comparison
            right_half_flipped = np.fliplr(right_half)
            
            # Calculate symmetry score (0-1)
            symmetry_diff = np.mean(np.abs(left_half - right_half_flipped))
            symmetry_score = 1.0 - min(symmetry_diff, 1.0)
            feedback["symmetry_score"] = float(symmetry_score)
            
            # Analyze completeness (coverage of image)
            coverage = np.sum(edges > 0) / (h * w)
            completeness_score = min(coverage * 2, 1.0)  # Scale up for better UX
            feedback["completeness_score"] = float(completeness_score)
            
            # Analyze clarity (edge strength)
            edge_strength = np.mean(edges) / 255.0
            clarity_score = min(edge_strength * 2, 1.0)
            feedback["clarity_score"] = float(clarity_score)
            
            # Generate feedback based on scores
            if symmetry_score < 0.6:
                feedback["issues"].append("Face not symmetric")
                feedback["suggestions"].append("Improve jawline definition and cheek symmetry")
            
            if completeness_score < 0.4:
                feedback["issues"].append("Incomplete sketch")
                feedback["suggestions"].append("Add more facial features (eyes, nose, mouth)")
            
            if clarity_score < 0.5:
                feedback["issues"].append("Sketch is too faint")
                feedback["suggestions"].append("Draw with more defined lines")
            
            # Detect eye alignment
            eye_region = image_array[h//4:h//3, :]
            eye_edges = cv2.Canny((eye_region * 255).astype(np.uint8), 30, 100)
            
            # Find horizontal line consistency in eye region
            horizontal_lines = np.sum(eye_edges, axis=1)
            if len(horizontal_lines) > 0:
                line_variance = np.var(horizontal_lines)
                if line_variance < 50:
                    feedback["issues"].append("Eyes not properly aligned")
                    feedback["suggestions"].append("Ensure eyes are at same horizontal level")
            
            # Calculate overall quality score
            overall = (symmetry_score + completeness_score + clarity_score) / 3.0
            feedback["overall_quality"] = float(overall)
            
            return feedback
        
        except Exception as e:
            print(f"Quality analysis error: {e}")
            return feedback
    
    def find_matches(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Find top-K matches in database."""
        try:
            distances, suspect_ids = self.faiss_indexer.search(embedding, k=k)
            
            matches = []
            for distance, suspect_id in zip(distances, suspect_ids):
                suspect = self.db_manager.get_suspect(int(suspect_id))
                if not suspect:
                    continue

                confidence = 1.0 / (1.0 + float(distance))  # Convert distance to confidence
                matches.append({
                    "id": suspect["id"],
                    "name": suspect.get("name", "Unknown"),
                    "confidence": float(confidence),
                    "distance": float(distance),
                    "image_url": suspect.get("image_url", "")
                })
            
            return matches
        except Exception as e:
            print(f"Match finding error: {e}")
            return []

    def _create_attention_map(self, image_array: np.ndarray) -> np.ndarray:
        """
        Create a lightweight attention proxy map from local gradients.
        This is a practical explainability overlay when true model attention is unavailable.
        """
        grad_x = cv2.Sobel((image_array * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel((image_array * 255).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.GaussianBlur(magnitude, (11, 11), 0)
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
    
    async def process_realtime_sketch(self, image_data: bytes) -> Dict:
        """
        Process sketch in real-time and return feedback + matches.
        This is called frequently (every 300-500ms) as user draws.
        """
        # Debounce check
        current_time = time.time() * 1000
        if current_time - self.last_feedback_time < self.feedback_debounce_ms:
            return {"status": "debounced"}
        
        self.last_feedback_time = current_time
        
        try:
            # Preprocess
            preprocessed, original = self.preprocess_sketch(image_data)
            if preprocessed is None:
                return {"status": "error", "message": "Failed to preprocess image"}
            
            # Extract features
            embedding = self.extract_features(preprocessed)
            if embedding is None:
                return {"status": "error", "message": "Failed to extract features"}
            
            # Analyze quality
            quality_feedback = self.analyze_sketch_quality(preprocessed)
            
            # Find matches
            matches = self.find_matches(embedding, k=3)  # Top-3 for real-time
            
            # Create edge map for visualization
            edges = cv2.Canny((preprocessed * 255).astype(np.uint8), 50, 150)
            edges_base64 = self._encode_image_base64(edges)
            
            # Create preprocessed image visualization
            preprocessed_vis = (preprocessed * 255).astype(np.uint8)
            preprocessed_base64 = self._encode_image_base64(preprocessed_vis)

            # Attention-like heatmap visualization
            attention_map = self._create_attention_map(preprocessed)
            attention_map_base64 = self._encode_image_base64(attention_map)
            
            # Get average inference time
            avg_inference_time = np.mean(list(self.inference_times)) if self.inference_times else 0
            
            return {
                "status": "success",
                "quality_feedback": quality_feedback,
                "matches": matches[:3],  # Top-3 for real-time
                "edge_map": edges_base64,
                "attention_map": attention_map_base64,
                "preprocessed_image": preprocessed_base64,
                "inference_time_ms": float(avg_inference_time * 1000),
                "timestamp": current_time
            }
        
        except Exception as e:
            print(f"Real-time processing error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _encode_image_base64(self, image_array: np.ndarray) -> str:
        """Encode numpy array as base64 PNG."""
        try:
            image = Image.fromarray(image_array)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Image encoding error: {e}")
            return ""
    
    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics."""
        return {
            "avg_inference_time_ms": float(np.mean(list(self.inference_times)) * 1000) if self.inference_times else 0,
            "min_inference_time_ms": float(np.min(list(self.inference_times)) * 1000) if self.inference_times else 0,
            "max_inference_time_ms": float(np.max(list(self.inference_times)) * 1000) if self.inference_times else 0,
            "recent_samples": len(self.inference_times)
        }


class SketchFeedbackWebSocket:
    """WebSocket handler for real-time sketch feedback."""
    
    def __init__(self, feedback_engine: RealtimeFeedbackEngine):
        self.feedback_engine = feedback_engine
        self.active_connections: List = []
    
    async def connect(self, websocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def disconnect(self, websocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
    
    async def process_sketch_frame(self, websocket, image_data: bytes):
        """Process sketch frame and send feedback."""
        feedback = await self.feedback_engine.process_realtime_sketch(image_data)
        await websocket.send_json(feedback)
