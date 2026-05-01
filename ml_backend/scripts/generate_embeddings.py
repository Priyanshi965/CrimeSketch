"""
Batch embedding generation script for all datasets.
Generates embeddings for all face images and indexes them in FAISS.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese_model import ModelManager
from preprocessing.image_preprocessor import ImagePreprocessor
from embeddings.faiss_indexer import FAISSIndexer
from database.schema import DatabaseManager
from utils.dataset_loader import DatasetLoader

# Configuration — paths relative to project root
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DIM = 512
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_BACKEND_DIR = os.path.dirname(_SCRIPTS_DIR)
_PROJECT_ROOT = os.path.dirname(_ML_BACKEND_DIR)
DB_PATH = os.path.join(_ML_BACKEND_DIR, "database", "crimesketch.db")
FAISS_INDEX_PATH = os.path.join(_ML_BACKEND_DIR, "embeddings", "index.faiss")
FAISS_METADATA_PATH = os.path.join(_ML_BACKEND_DIR, "embeddings", "metadata.pkl")
DATASET_BASE_PATH = os.path.join(_PROJECT_ROOT, "datasets", "organized")

def generate_embeddings_for_dataset(model_manager, preprocessor, dataset_path, dataset_name, db_manager, faiss_indexer):
    """Generate embeddings for all images in a dataset."""
    
    embeddings_list = []
    suspect_ids_list = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"\n[{dataset_name}] Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"[{dataset_name}] No images found, skipping...")
        return 0
    
    # Process each image
    processed_count = 0
    for idx, image_path in enumerate(tqdm(image_files, desc=f"Generating embeddings for {dataset_name}")):
        try:
            # Preprocess image (returns tuple: preprocessed_array, metadata)
            preprocessed, metadata = preprocessor.preprocess(image_path)
            
            # Convert (H, W) grayscale to (1, 3, H, W) tensor
            preprocessed_tensor = torch.from_numpy(preprocessed).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
            
            # Get embedding
            embedding = model_manager.get_embedding(preprocessed_tensor)
            
            # Extract filename for suspect name
            filename = Path(image_path).stem
            
            # Add suspect to database
            suspect_id = db_manager.add_suspect(
                name=filename,
                image_path=image_path,
                age=None,
                gender=None,
                city=None,
                crime_type=None,
                risk_level="medium",
                dataset_source=dataset_name
            )
            
            # Add embedding to database
            db_manager.add_embedding(suspect_id, embedding[0])
            
            # Collect for batch FAISS indexing
            embeddings_list.append(embedding[0])
            suspect_ids_list.append(suspect_id)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"[{dataset_name}] Successfully processed {processed_count} images")
    
    # Add batch to FAISS index
    if len(embeddings_list) > 0:
        embeddings_array = np.array(embeddings_list)
        faiss_indexer.add_embeddings(embeddings_array, suspect_ids_list)
        print(f"[{dataset_name}] Added {len(embeddings_list)} embeddings to FAISS index")
    
    return processed_count


def main():
    """Main function to generate embeddings for all datasets."""
    
    print("=" * 80)
    print("CrimeSketch AI - Batch Embedding Generation")
    print("=" * 80)
    
    # Initialize components
    print("\nInitializing components...")
    
    # Initialize database
    db_manager = DatabaseManager(DB_PATH)
    print(f"✓ Database initialized: {DB_PATH}")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(160, 160), use_face_alignment=False)
    print(f"✓ Preprocessor initialized")
    
    # Initialize model (load trained checkpoint if available)
    _model_path = os.path.join(_ML_BACKEND_DIR, "models", "siamese_checkpoint.pt")
    model_manager = ModelManager(
        model_type='siamese',
        embedding_dim=EMBEDDING_DIM,
        device=DEVICE,
        model_path=_model_path if os.path.exists(_model_path) else None
    )
    if os.path.exists(_model_path):
        print(f"✓ Model initialized on {DEVICE} (loaded checkpoint: {_model_path})")
    else:
        print(f"⚠ Model initialized on {DEVICE} (no checkpoint found — using untrained weights)")
    
    # Initialize FAISS indexer
    faiss_indexer = FAISSIndexer(embedding_dim=EMBEDDING_DIM, index_type='flat')
    print(f"✓ FAISS indexer initialized")
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader()
    print(f"✓ Dataset loader initialized")
    
    print("\n" + "=" * 80)
    print("Starting embedding generation...")
    print("=" * 80)
    
    total_processed = 0
    
    # Only index photos (not sketches) — sketches are queries, photos are the gallery
    # Process Dataset 1 photos
    dataset1_photos_path = os.path.join(DATASET_BASE_PATH, "dataset1", "photos")
    dataset1_path = os.path.join(DATASET_BASE_PATH, "dataset1")
    _d1_path = dataset1_photos_path if os.path.exists(dataset1_photos_path) else dataset1_path
    if os.path.exists(_d1_path):
        count = generate_embeddings_for_dataset(
            model_manager, preprocessor, _d1_path,
            "Dataset1", db_manager, faiss_indexer
        )
        total_processed += count
    else:
        print(f"⚠ Dataset1 path not found: {_d1_path}")

    # Process Dataset 2 photos
    dataset2_photos_path = os.path.join(DATASET_BASE_PATH, "dataset2", "photos")
    dataset2_path = os.path.join(DATASET_BASE_PATH, "dataset2")
    _d2_path = dataset2_photos_path if os.path.exists(dataset2_photos_path) else dataset2_path
    if os.path.exists(_d2_path):
        count = generate_embeddings_for_dataset(
            model_manager, preprocessor, _d2_path,
            "Dataset2", db_manager, faiss_indexer
        )
        total_processed += count
    else:
        print(f"⚠ Dataset2 path not found: {_d2_path}")

    # Process Dataset 3 (CelebA)
    dataset3_path = os.path.join(DATASET_BASE_PATH, "dataset3")
    if os.path.exists(dataset3_path):
        count = generate_embeddings_for_dataset(
            model_manager, preprocessor, dataset3_path, 
            "Dataset3", db_manager, faiss_indexer
        )
        total_processed += count
    else:
        print(f"⚠ Dataset3 path not found: {dataset3_path}")
    
    print("\n" + "=" * 80)
    print("Saving FAISS index...")
    print("=" * 80)
    
    # Save FAISS index
    faiss_indexer.save_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    print(f"✓ FAISS index saved: {FAISS_INDEX_PATH}")
    print(f"✓ FAISS metadata saved: {FAISS_METADATA_PATH}")
    
    # Get and display statistics
    print("\n" + "=" * 80)
    print("Embedding Generation Complete!")
    print("=" * 80)
    
    stats = db_manager.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total Suspects: {stats['total_suspects']}")
    print(f"  Total Embeddings: {stats['total_embeddings']}")
    print(f"  Average Confidence: {stats['avg_match_confidence']:.2f}")
    print(f"\nTotal Images Processed: {total_processed}")
    
    # Close database
    db_manager.close()
    print("\n✓ Database connection closed")
    print("\nEmbedding generation completed successfully!")


if __name__ == "__main__":
    main()
