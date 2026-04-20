"""
FastAPI Backend for CrimeSketch AI

Provides REST endpoints for sketch-to-face matching, suspect management,
and FAISS indexing operations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import io
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image
import time
import uuid
from datetime import datetime
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_backend'))

from preprocessing.image_preprocessor import ImagePreprocessor
from models.siamese_model import ModelManager
from embeddings.faiss_indexer import FAISSIndexer, SimilarityMatcher
from database.schema import DatabaseManager
from utils.dataset_loader import DatasetLoader
from training.pipeline import run_training_pipeline

# Initialize FastAPI app
app = FastAPI(title="CrimeSketch AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
db_manager = None
model_manager = None
faiss_indexer = None
preprocessor = None
dataset_loader = None
training_state = {
    "status": "idle",
    "last_started_at": None,
    "last_completed_at": None,
    "last_config": None,
    "last_report": None,
}


def _build_training_attempts(config: Dict[str, Any], max_attempts: int) -> List[Dict[str, Any]]:
    """Create a bounded set of progressively stronger configs for auto-tuning."""
    attempts: List[Dict[str, Any]] = []

    base = dict(config)
    attempts.append(base)

    epoch_candidates = [
        max(base["epochs"], 20),
        max(base["epochs"], 30),
        max(base["epochs"], 40),
        max(base["epochs"], 60),
    ]
    lr_candidates = [base["learning_rate"], 5e-5, 3e-5]
    batch_candidates = [base["batch_size"], 16, 8]
    loss_candidates = [base["loss_type"], "triplet", "contrastive"]

    for loss in loss_candidates:
        for epochs in epoch_candidates:
            for lr in lr_candidates:
                for batch in batch_candidates:
                    candidate = dict(base)
                    candidate["loss_type"] = loss
                    candidate["epochs"] = int(epochs)
                    candidate["learning_rate"] = float(lr)
                    candidate["batch_size"] = int(batch)
                    attempts.append(candidate)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in attempts:
        key = (
            item["loss_type"],
            item["epochs"],
            item["batch_size"],
            round(float(item["learning_rate"]), 9),
            item["target_accuracy"],
            item["gate_metric"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max(1, max_attempts):
            break

    return deduped

# Configuration — paths relative to this file's location
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)
MODEL_PATH = os.path.join(_PROJECT_ROOT, "ml_backend", "models", "siamese_checkpoint.pt")
FAISS_INDEX_PATH = os.path.join(_PROJECT_ROOT, "ml_backend", "embeddings", "index.faiss")
FAISS_METADATA_PATH = os.path.join(_PROJECT_ROOT, "ml_backend", "embeddings", "metadata.pkl")
DB_PATH = os.path.join(_PROJECT_ROOT, "ml_backend", "database", "crimesketch.db")
UPLOADS_PATH = os.path.join(_PROJECT_ROOT, "datasets", "uploads")

# Pydantic models
class PredictRequest(BaseModel):
    image_data: str  # base64 encoded image
    top_k: int = 5

class TrainRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.0001
    loss_type: str = "triplet"
    target_accuracy: float = 0.94
    top_k: int = 5
    seed: int = 42
    reindex_after_train: bool = True
    max_train_samples: int = 0
    max_eval_samples: int = 0
    gate_metric: str = "accuracy"
    resume_from_checkpoint: bool = True
    triplet_margin: float = 0.5
    use_augmentation: bool = True
    batch_hard_triplet: bool = True
    auto_tune: bool = True
    max_attempts: int = 6

class AddSuspectRequest(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    crime_type: Optional[str] = None
    risk_level: str = "medium"
    dataset_source: str = "unknown"
    image_data: str  # base64 encoded image

class SuspectInfo(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    crime_type: Optional[str] = None
    risk_level: str = "medium"
    dataset_source: str = "unknown"

class MatchResult(BaseModel):
    suspect_id: int
    name: str
    age: Optional[int]
    gender: Optional[str]
    city: Optional[str]
    crime_type: Optional[str]
    risk_level: str
    image_url: Optional[str]
    distance: float
    confidence: float
    confidence_percentage: float

class PredictionResponse(BaseModel):
    best_match: Optional[MatchResult]
    top_k_matches: List[MatchResult]
    search_time_ms: float
    pipeline_steps: Dict[str, Any]

class StatsResponse(BaseModel):
    total_suspects: int
    total_embeddings: int
    avg_match_confidence: float
    dataset_stats: Dict[str, int]

class SearchHistoryItem(BaseModel):
    id: int
    best_match_id: int
    best_match_score: float
    search_time_ms: float
    created_at: str


@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup."""
    global db_manager, model_manager, faiss_indexer, preprocessor, dataset_loader
    
    print("Initializing CrimeSketch AI backend...")
    
    # Initialize database
    db_manager = DatabaseManager(DB_PATH)
    print("Database initialized")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(160, 160), use_face_alignment=False)
    print("Preprocessor initialized")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_manager = ModelManager(
        model_type='siamese',
        embedding_dim=512,
        device=device,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None
    )
    print(f"Model initialized on {device}")
    
    # Initialize FAISS indexer
    faiss_indexer = FAISSIndexer(embedding_dim=512, index_type='flat')
    
    # Try to load existing index
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_indexer.load_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        print("FAISS index loaded")
    else:
        print("No existing FAISS index found")
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader()
    print("Dataset loader initialized")
    
    print("Backend initialization complete!")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager is not None,
        "database_ready": db_manager is not None,
        "faiss_ready": faiss_indexer is not None
    }


@app.get("/image/{suspect_id}")
async def get_suspect_image(suspect_id: int):
    """Serve the photo for a suspect directly from disk."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    suspect = db_manager.get_suspect(suspect_id)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found")
    image_path = suspect.get("image_path")
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(image_path, media_type="image/jpeg")


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictionResponse:
    """
    Predict matches for uploaded sketch (base64 encoded).

    Args:
        request: JSON body with base64 image_data and optional top_k

    Returns:
        Top-K matches with confidence scores
    """
    if not all([db_manager, model_manager, faiss_indexer, preprocessor]):
        raise HTTPException(status_code=503, detail="Backend not initialized")

    start_time = time.time()
    temp_path = None

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Write to a proper cross-platform temp file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, image)

        # Preprocess image
        preprocessed, metadata = preprocessor.preprocess(temp_path)

        # Convert (H, W) grayscale to (1, 3, H, W) tensor
        preprocessed_tensor = torch.from_numpy(preprocessed).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()

        embedding = model_manager.get_embedding(preprocessed_tensor)

        # Search FAISS index
        k = min(request.top_k, 10)
        distances, matched_suspect_ids = faiss_indexer.search(embedding, k=k)

        # Get suspect information
        suspect_results = []
        for distance, suspect_id in zip(distances, matched_suspect_ids):
            suspect_info = db_manager.get_suspect(suspect_id)
            if suspect_info:
                result = {
                    'suspect_id': suspect_id,
                    'name': suspect_info['name'],
                    'age': suspect_info['age'],
                    'gender': suspect_info['gender'],
                    'city': suspect_info['city'],
                    'crime_type': suspect_info['crime_type'],
                    'risk_level': suspect_info['risk_level'],
                    'image_url': f"/image/{suspect_id}",
                    'distance': float(distance),
                    'confidence': SimilarityMatcher.distance_to_confidence(distance),
                    'confidence_percentage': round(SimilarityMatcher.distance_to_confidence(distance) * 100, 2)
                }
                suspect_results.append(result)

        best_match = suspect_results[0] if suspect_results else None
        search_time_ms = (time.time() - start_time) * 1000

        if best_match:
            db_manager.log_search(
                query_sketch_path=temp_path,
                top_k_results=suspect_results,
                best_match_id=best_match['suspect_id'],
                best_match_score=best_match['confidence'],
                search_time_ms=search_time_ms
            )

        return PredictionResponse(
            best_match=MatchResult(**best_match) if best_match else None,
            top_k_matches=[MatchResult(**r) for r in suspect_results],
            search_time_ms=search_time_ms,
            pipeline_steps={
                'preprocessing_steps': list(metadata['steps'].keys()),
                'statistics': metadata['statistics']
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/add_suspect")
async def add_suspect(request: AddSuspectRequest) -> Dict[str, Any]:
    """
    Add a new suspect to the database.

    Args:
        request: JSON body with base64 image_data and suspect metadata

    Returns:
        Suspect ID and confirmation
    """
    if not all([db_manager, model_manager, preprocessor]):
        raise HTTPException(status_code=503, detail="Backend not initialized")

    temp_path = None
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Save to uploads directory
        os.makedirs(UPLOADS_PATH, exist_ok=True)
        image_path = os.path.join(UPLOADS_PATH, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(image_path, image)
        temp_path = image_path

        # Preprocess and get embedding
        preprocessed, _ = preprocessor.preprocess(image_path)
        preprocessed_tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0)
        preprocessed_tensor = preprocessed_tensor.repeat(1, 3, 1, 1).float()

        embedding = model_manager.get_embedding(preprocessed_tensor)

        # Add to database
        suspect_id = db_manager.add_suspect(
            name=request.name,
            image_path=image_path,
            age=request.age,
            gender=request.gender,
            city=request.city,
            crime_type=request.crime_type,
            risk_level=request.risk_level,
            dataset_source=request.dataset_source or "manual_upload"
        )

        # Add embedding and update FAISS index
        db_manager.add_embedding(suspect_id, embedding[0])
        faiss_indexer.add_embeddings(embedding, [suspect_id])

        return {
            "success": True,
            "suspect_id": suspect_id,
            "message": "Suspect added successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suspects")
async def get_suspects(city: Optional[str] = None, crime_type: Optional[str] = None,
                      risk_level: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Get suspects with optional filtering.
    
    Args:
        city: Filter by city
        crime_type: Filter by crime type
        risk_level: Filter by risk level
        limit: Maximum number of results
        
    Returns:
        List of suspects matching criteria
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        suspects = db_manager.get_suspects_filtered(
            city=city,
            crime_type=crime_type,
            risk_level=risk_level,
            limit=limit
        )
        # Inject served image URLs so the frontend can display photos
        for s in suspects:
            s["image_url"] = f"/image/{s['id']}"
        return {"suspects": suspects, "total": len(suspects)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/re_index")
async def re_index(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Rebuild FAISS index from database.
    
    Returns:
        Status of re-indexing operation
    """
    if not all([db_manager, faiss_indexer]):
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    def rebuild_index():
        embeddings, suspect_ids = db_manager.get_all_embeddings()
        faiss_indexer.reset()
        if len(embeddings) > 0:
            faiss_indexer.add_embeddings(embeddings, suspect_ids)
            faiss_indexer.save_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        print(f"Index rebuilt with {len(suspect_ids)} embeddings")
    
    background_tasks.add_task(rebuild_index)
    
    return {
        "status": "re-indexing",
        "message": "Index re-indexing started in background"
    }


@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Start model training in the background.
    """
    global training_state

    if training_state["status"] == "running":
        return {
            "status": "already_running",
            "message": "Training is already in progress.",
            "training_state": training_state
        }

    def run_training_job():
        global training_state
        training_state["status"] = "running"
        training_state["last_started_at"] = datetime.now().isoformat()
        training_state["last_config"] = request.model_dump()
        training_state["last_report"] = None

        try:
            request_config = request.model_dump()
            pipeline_config = {
                "epochs": request_config["epochs"],
                "batch_size": request_config["batch_size"],
                "learning_rate": request_config["learning_rate"],
                "loss_type": request_config["loss_type"],
                "target_accuracy": request_config["target_accuracy"],
                "top_k": request_config["top_k"],
                "seed": request_config["seed"],
                "reindex_after_train": request_config["reindex_after_train"],
                "max_train_samples": request_config["max_train_samples"],
                "max_eval_samples": request_config["max_eval_samples"],
                "gate_metric": request_config["gate_metric"],
                "resume_from_checkpoint": request_config["resume_from_checkpoint"],
                "triplet_margin": request_config["triplet_margin"],
                "use_augmentation": request_config["use_augmentation"],
                "batch_hard_triplet": request_config["batch_hard_triplet"],
            }

            if request.auto_tune:
                attempts = _build_training_attempts(pipeline_config, request.max_attempts)
            else:
                attempts = [pipeline_config]

            attempt_reports = []
            best_report = None

            for attempt_number, attempt_cfg in enumerate(attempts, start=1):
                print(f"[train] attempt {attempt_number}/{len(attempts)} cfg={attempt_cfg}", flush=True)
                report = run_training_pipeline(project_root=_PROJECT_ROOT, config=attempt_cfg)
                attempt_reports.append({
                    "attempt": attempt_number,
                    "config": attempt_cfg,
                    "status": report.get("status"),
                    "gate": report.get("gate"),
                    "test": report.get("test"),
                    "validation": report.get("validation"),
                })

                if best_report is None:
                    best_report = report
                else:
                    prev = float(best_report.get("gate", {}).get("actual", 0.0))
                    curr = float(report.get("gate", {}).get("actual", 0.0))
                    if curr > prev:
                        best_report = report

                if report.get("gate", {}).get("passed"):
                    best_report = report
                    break

            # Reload model and index so serving path uses latest artifacts immediately.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            refreshed_model = ModelManager(
                model_type='siamese',
                embedding_dim=512,
                device=device,
                model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None
            )
            refreshed_index = FAISSIndexer(embedding_dim=512, index_type='flat')
            if os.path.exists(FAISS_INDEX_PATH):
                refreshed_index.load_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)

            globals()["model_manager"] = refreshed_model
            globals()["faiss_indexer"] = refreshed_index

            passed = bool(best_report and best_report.get("gate", {}).get("passed"))
            training_state["status"] = "completed" if passed else "failed_target"
            training_state["last_report"] = {
                "attempts": attempt_reports,
                "best_attempt": best_report,
                "gate": best_report.get("gate") if best_report else None,
            }
        except Exception as e:
            training_state["status"] = "failed"
            training_state["last_report"] = {
                "status": "failed",
                "error": str(e),
            }

        training_state["last_completed_at"] = datetime.now().isoformat()

    background_tasks.add_task(run_training_job)

    return {
        "status": "started",
        "message": "Training job has been started in the background.",
        "config": request.model_dump(),
        "note": "Training runs with a 94% target gate and optional auto-tuning retries."
    }


@app.get("/train/status")
async def train_status() -> Dict[str, Any]:
    """Get training job status and latest report."""
    return training_state


@app.get("/stats")
async def get_stats() -> StatsResponse:
    """
    Get dataset and system statistics.
    
    Returns:
        Statistics dictionary
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    db_stats = db_manager.get_stats()
    dataset_stats = dataset_loader.get_dataset_stats() if dataset_loader else {}
    
    return StatsResponse(
        total_suspects=db_stats['total_suspects'],
        total_embeddings=db_stats['total_embeddings'],
        avg_match_confidence=db_stats['avg_match_confidence'],
        dataset_stats=dataset_stats
    )


@app.get("/search_history")
async def get_search_history(limit: int = 100, session_id: Optional[str] = None) -> List[SearchHistoryItem]:
    """
    Get search history.
    
    Args:
        limit: Maximum number of records
        session_id: Filter by session ID
        
    Returns:
        List of search history items
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    history = db_manager.get_search_history(limit, session_id)
    return [SearchHistoryItem(**item) for item in history]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
