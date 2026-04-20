"""
FastAPI endpoints for real-time sketch feedback system.
Integrates with ml_realtime.py for low-latency inference.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import asyncio
from typing import Optional
import logging

# Import the realtime feedback engine
from ml_realtime import RealtimeFeedbackEngine, SketchFeedbackWebSocket

logger = logging.getLogger(__name__)

def create_realtime_app(
    model_manager,
    faiss_indexer,
    db_manager,
    base_app: Optional[FastAPI] = None
) -> FastAPI:
    """
    Create FastAPI app with real-time feedback endpoints.
    
    Args:
        model_manager: ML model manager for embeddings
        faiss_indexer: FAISS index for similarity search
        db_manager: Database manager for suspect data
        base_app: Optional existing FastAPI app to extend
    
    Returns:
        FastAPI app with real-time endpoints
    """
    
    app = base_app or FastAPI(title="CrimeSketch AI - Real-Time Feedback")
    
    # Initialize feedback engine
    feedback_engine = RealtimeFeedbackEngine(
        model_manager=model_manager,
        faiss_indexer=faiss_indexer,
        db_manager=db_manager,
        cache_size=20
    )
    
    # Initialize WebSocket manager
    ws_manager = SketchFeedbackWebSocket(feedback_engine)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ============================================================================
    # REAL-TIME FEEDBACK ENDPOINTS
    # ============================================================================
    
    @app.post("/realtime-feedback")
    async def realtime_feedback(data: dict):
        """
        Process sketch frame and return real-time feedback.
        
        Request body:
        {
            "image_data": "base64_encoded_image"
        }
        
        Response:
        {
            "status": "success|error|debounced",
            "quality_feedback": {
                "symmetry_score": 0.0-1.0,
                "completeness_score": 0.0-1.0,
                "clarity_score": 0.0-1.0,
                "overall_quality": 0.0-1.0,
                "issues": ["issue1", "issue2"],
                "suggestions": ["suggestion1", "suggestion2"]
            },
            "matches": [
                {
                    "id": 123,
                    "name": "John Doe",
                    "confidence": 0.95,
                    "distance": 0.05,
                    "image_url": "..."
                }
            ],
            "edge_map": "base64_encoded_image",
            "preprocessed_image": "base64_encoded_image",
            "inference_time_ms": 45.2
        }
        """
        try:
            image_data_b64 = data.get("image_data")
            if not image_data_b64:
                raise ValueError("Missing image_data")
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data_b64)
            
            # Process sketch
            result = await feedback_engine.process_realtime_sketch(image_bytes)
            
            return JSONResponse(content=result)
        
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process sketch")
    
    @app.websocket("/ws/sketch-feedback")
    async def websocket_sketch_feedback(websocket: WebSocket):
        """
        WebSocket endpoint for real-time sketch feedback streaming.
        
        Client sends: base64 encoded image frames
        Server sends: feedback + matches in real-time
        """
        await ws_manager.connect(websocket)
        try:
            while True:
                # Receive image data from client
                data = await websocket.receive_json()
                image_data_b64 = data.get("image_data")
                
                if not image_data_b64:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Missing image_data"
                    })
                    continue
                
                # Decode and process
                try:
                    image_bytes = base64.b64decode(image_data_b64)
                    result = await feedback_engine.process_realtime_sketch(image_bytes)
                    await websocket.send_json(result)
                except Exception as e:
                    logger.error(f"WebSocket processing error: {e}")
                    await websocket.send_json({
                        "status": "error",
                        "message": str(e)
                    })
        
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await ws_manager.disconnect(websocket)
    
    @app.get("/performance-metrics")
    async def get_performance_metrics():
        """
        Get real-time system performance metrics.
        
        Response:
        {
            "avg_inference_time_ms": 45.2,
            "min_inference_time_ms": 32.1,
            "max_inference_time_ms": 78.5,
            "recent_samples": 100
        }
        """
        try:
            metrics = feedback_engine.get_performance_metrics()
            return JSONResponse(content=metrics)
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get metrics")
    
    @app.post("/batch-realtime-feedback")
    async def batch_realtime_feedback(data: dict):
        """
        Process multiple sketch frames in batch for comparison.
        Useful for analyzing sketch evolution over time.
        
        Request body:
        {
            "frames": [
                {"image_data": "base64_1", "timestamp": 0},
                {"image_data": "base64_2", "timestamp": 300},
                {"image_data": "base64_3", "timestamp": 600}
            ]
        }
        
        Response:
        {
            "status": "success",
            "frames": [
                {
                    "timestamp": 0,
                    "quality_feedback": {...},
                    "matches": [...]
                },
                ...
            ],
            "quality_trend": {
                "symmetry_trend": [0.3, 0.5, 0.7],
                "completeness_trend": [0.2, 0.4, 0.6],
                "clarity_trend": [0.4, 0.6, 0.8]
            }
        }
        """
        try:
            frames = data.get("frames", [])
            if not frames:
                raise ValueError("No frames provided")
            
            results = []
            symmetry_trend = []
            completeness_trend = []
            clarity_trend = []
            
            for frame in frames:
                image_data_b64 = frame.get("image_data")
                timestamp = frame.get("timestamp", 0)
                
                if not image_data_b64:
                    continue
                
                image_bytes = base64.b64decode(image_data_b64)
                result = await feedback_engine.process_realtime_sketch(image_bytes)
                
                if result["status"] == "success":
                    feedback = result.get("quality_feedback", {})
                    results.append({
                        "timestamp": timestamp,
                        "quality_feedback": feedback,
                        "matches": result.get("matches", [])
                    })
                    
                    symmetry_trend.append(feedback.get("symmetry_score", 0))
                    completeness_trend.append(feedback.get("completeness_score", 0))
                    clarity_trend.append(feedback.get("clarity_score", 0))
            
            return JSONResponse(content={
                "status": "success",
                "frames": results,
                "quality_trend": {
                    "symmetry_trend": symmetry_trend,
                    "completeness_trend": completeness_trend,
                    "clarity_trend": clarity_trend
                }
            })
        
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process batch")
    
    @app.get("/realtime-config")
    async def get_realtime_config():
        """
        Get real-time feedback configuration.
        
        Response:
        {
            "debounce_ms": 300,
            "cache_size": 20,
            "max_inference_time_ms": 100,
            "supported_formats": ["png", "jpg", "jpeg"]
        }
        """
        return JSONResponse(content={
            "debounce_ms": feedback_engine.feedback_debounce_ms,
            "cache_size": feedback_engine.embedding_cache.maxlen,
            "max_inference_time_ms": 100,
            "supported_formats": ["png", "jpg", "jpeg"],
            "quality_thresholds": {
                "symmetry_warning": 0.6,
                "completeness_warning": 0.4,
                "clarity_warning": 0.5
            }
        })
    
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn
    from ml_backend.models.siamese_model import ModelManager
    from ml_backend.embeddings.faiss_indexer import FAISSIndexer
    from ml_backend.database.schema import DatabaseManager
    
    # Initialize components
    model_manager = ModelManager(model_path="ml_backend/models/siamese_model.pth")
    faiss_indexer = FAISSIndexer(index_path="ml_backend/embeddings/index.faiss")
    db_manager = DatabaseManager(db_path="ml_backend/database/crimesketch.db")
    
    # Create app
    app = create_realtime_app(model_manager, faiss_indexer, db_manager)
    
    # Run
    uvicorn.run(app, host="0.0.0.0", port=8001)
