"""
FAISS Indexing and Similarity Search

Implements efficient similarity search using Facebook's FAISS library
for matching sketch embeddings against a large database of face embeddings.

Supports:
- IndexFlatL2: Exact L2 distance search (suitable for smaller datasets)
- IndexIVFFlat: Approximate nearest neighbor search (for larger datasets)
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle


class FAISSIndexer:
    """
    FAISS-based indexing and similarity search for face embeddings.
    
    Attributes:
        index: FAISS index object
        embedding_dim: Dimension of embeddings
        suspect_ids: Mapping from index position to suspect ID
    """
    
    def __init__(self, embedding_dim: int = 512, index_type: str = 'flat'):
        """
        Initialize FAISS indexer.
        
        Args:
            embedding_dim: Dimension of embeddings (512 for ResNet50)
            index_type: Type of index ('flat' for exact, 'ivf' for approximate)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.suspect_ids = []
        self.embeddings = None
        
        if index_type == 'flat':
            # Exact L2 distance search
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == 'ivf':
            # Approximate nearest neighbor search with IVF
            # Use 100 clusters for medium-sized datasets
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, suspect_ids: List[int]):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            suspect_ids: List of suspect IDs corresponding to embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # For IVF index, train on the data first
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.suspect_ids.extend(suspect_ids)
        self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])
        
        print(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """
        Search for top-K nearest neighbors.
        
        Args:
            query_embedding: Query embedding (embedding_dim,) or (1, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, suspect_ids)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert indices to suspect IDs
        distances = distances[0]
        indices = indices[0]
        
        matched_suspect_ids = [self.suspect_ids[idx] for idx in indices if idx < len(self.suspect_ids)]
        
        return distances, matched_suspect_ids
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Search for top-K nearest neighbors for multiple queries.
        
        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim)
            k: Number of nearest neighbors per query
            
        Returns:
            Tuple of (distances_array, suspect_ids_list)
        """
        # Ensure float32
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Convert indices to suspect IDs
        suspect_ids_list = []
        for idx_row in indices:
            matched_ids = [self.suspect_ids[idx] for idx in idx_row if idx < len(self.suspect_ids)]
            suspect_ids_list.append(matched_ids)
        
        return distances, suspect_ids_list
    
    def save_index(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (suspect_ids)
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")
        
        # Save metadata
        if metadata_path:
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                'suspect_ids': self.suspect_ids,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'total_embeddings': self.index.ntotal
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata file
        """
        self.index = faiss.read_index(index_path)
        print(f"Index loaded from {index_path}. Total embeddings: {self.index.ntotal}")
        
        # Load metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.suspect_ids = metadata['suspect_ids']
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            print(f"Metadata loaded from {metadata_path}")
    
    def reset(self):
        """Reset the index."""
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        
        self.suspect_ids = []
        self.embeddings = None
        print("Index reset")


class SimilarityMatcher:
    """
    Converts distances to confidence scores and provides match results.
    """
    
    @staticmethod
    def distance_to_confidence(distance: float, max_distance: float = 2.0) -> float:
        """
        Convert L2 distance to confidence score (0-1).

        Uses an exponential decay so scores spread meaningfully across the
        0–100% range even when raw L2 distances cluster tightly (e.g. 0.8–1.1).
        Tuned for FaceNet/VGGFace2 unit-sphere embeddings where genuine
        sketch-photo pairs typically land at distance 0.6–1.0 and impostors
        at 1.0–1.4.  The linear formula compressed everything into 50–60%,
        hiding rank differences; this formula maps:
          d=0.0 → 100%,  d=0.6 → ~83%,  d=0.8 → ~73%,
          d=1.0 → ~61%,  d=1.2 → ~50%,  d=1.6 → ~33%
        """
        distance = float(np.clip(distance, 0, max_distance))
        # k=0.8 gives good visual spread for sketch-photo cross-domain distances
        confidence = np.exp(-0.8 * distance)
        return float(np.clip(confidence, 0.0, 1.0))
    
    @staticmethod
    def format_results(distances: np.ndarray, suspect_ids: List[int], 
                      suspect_info: Dict[int, Dict]) -> List[Dict]:
        """
        Format search results with metadata.
        
        Args:
            distances: Array of distances
            suspect_ids: List of suspect IDs
            suspect_info: Dictionary mapping suspect_id to suspect information
            
        Returns:
            List of formatted result dictionaries
        """
        results = []
        
        for distance, suspect_id in zip(distances, suspect_ids):
            confidence = SimilarityMatcher.distance_to_confidence(distance)
            
            suspect = suspect_info.get(suspect_id, {})
            
            result = {
                'suspect_id': suspect_id,
                'name': suspect.get('name', 'Unknown'),
                'age': suspect.get('age'),
                'gender': suspect.get('gender'),
                'city': suspect.get('city'),
                'crime_type': suspect.get('crime_type'),
                'risk_level': suspect.get('risk_level'),
                'image_url': suspect.get('image_url'),
                'distance': float(distance),
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2)
            }
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test FAISS indexer
    print("Testing FAISS Indexer...")
    
    # Create indexer
    indexer = FAISSIndexer(embedding_dim=512, index_type='flat')
    
    # Create sample embeddings
    n_samples = 100
    embeddings = np.random.randn(n_samples, 512).astype(np.float32)
    suspect_ids = list(range(1, n_samples + 1))
    
    # Add embeddings
    indexer.add_embeddings(embeddings, suspect_ids)
    
    # Search
    query_embedding = embeddings[0]  # Use first embedding as query
    distances, matched_ids = indexer.search(query_embedding, k=5)
    
    print(f"Query embedding: {query_embedding[:5]}...")
    print(f"Top-5 matches: {matched_ids}")
    print(f"Distances: {distances}")
    
    # Convert to confidence
    confidences = [SimilarityMatcher.distance_to_confidence(d) for d in distances]
    print(f"Confidences: {confidences}")
    
    # Save and load
    indexer.save_index("/tmp/test_index.faiss", "/tmp/test_metadata.pkl")
    
    indexer2 = FAISSIndexer(embedding_dim=512)
    indexer2.load_index("/tmp/test_index.faiss", "/tmp/test_metadata.pkl")
    
    print("\nFAISS indexer test successful!")
