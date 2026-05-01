"""
Performance Optimization for Real-Time Inference
Implements caching, quantization, and batch processing strategies.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import threading
from functools import lru_cache
import hashlib

class EmbeddingCache:
    """LRU cache for embeddings with thread safety."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _hash_image(self, image_array: np.ndarray) -> str:
        """Create hash of image for cache key."""
        return hashlib.md5(image_array.tobytes()).hexdigest()
    
    def get(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._hash_image(image_array)
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, image_array: np.ndarray, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_image(image_array)
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                
                self.cache[key] = embedding
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class BatchProcessor:
    """Process multiple sketches in parallel for better GPU utilization."""
    
    def __init__(self, batch_size: int = 8, timeout_ms: int = 100):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.queue = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def add_to_batch(self, image_array: np.ndarray) -> None:
        """Add image to batch queue."""
        with self.condition:
            self.queue.append(image_array)
            self.condition.notify_all()
    
    def get_batch(self) -> Optional[List[np.ndarray]]:
        """Get batch when ready or timeout."""
        with self.condition:
            start_time = time.time()
            
            while len(self.queue) < self.batch_size:
                elapsed = (time.time() - start_time) * 1000
                remaining = self.timeout_ms - elapsed
                
                if remaining <= 0:
                    # Timeout - return partial batch
                    if self.queue:
                        batch = self.queue[:]
                        self.queue.clear()
                        return batch
                    return None
                
                self.condition.wait(timeout=remaining / 1000)
            
            # Full batch ready
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            return batch


class ModelQuantizer:
    """Quantize model for faster inference with minimal accuracy loss."""
    
    @staticmethod
    def quantize_embedding(embedding: np.ndarray, bits: int = 8) -> np.ndarray:
        """Quantize embedding from float32 to lower precision."""
        if bits == 8:
            # Scale to 0-255 range
            min_val = embedding.min()
            max_val = embedding.max()
            scaled = ((embedding - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return scaled
        
        elif bits == 16:
            return embedding.astype(np.float16)
        
        return embedding
    
    @staticmethod
    def dequantize_embedding(quantized: np.ndarray, original_range: Tuple[float, float]) -> np.ndarray:
        """Restore quantized embedding to float32."""
        min_val, max_val = original_range
        
        if quantized.dtype == np.uint8:
            return (quantized.astype(np.float32) / 255) * (max_val - min_val) + min_val
        
        return quantized.astype(np.float32)


class InferenceOptimizer:
    """Main optimizer coordinating all optimization strategies."""
    
    def __init__(
        self,
        model,
        cache_size: int = 1000,
        batch_size: int = 8,
        use_gpu: bool = True,
        quantize_embeddings: bool = False
    ):
        self.model = model
        self.cache = EmbeddingCache(max_size=cache_size)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.quantize_embeddings = quantize_embeddings
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Move model to GPU if available
        if self.use_gpu:
            self.model = self.model.cuda()
            self.model.eval()
    
    def optimize_inference(self, image_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimized inference with caching and quantization.
        
        Returns: (embedding, inference_time_ms)
        """
        start_time = time.time()
        
        # Check cache first
        cached_embedding = self.cache.get(image_array)
        if cached_embedding is not None:
            inference_time = (time.time() - start_time) * 1000
            return cached_embedding, inference_time
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float()
        
        if self.use_gpu:
            image_tensor = image_tensor.cuda()
        
        # Add batch dimension if needed
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        # Convert back to numpy
        embedding = embedding.cpu().numpy()
        
        # Quantize if enabled
        if self.quantize_embeddings:
            embedding = ModelQuantizer.quantize_embedding(embedding, bits=8)
        
        # Cache result
        self.cache.put(image_array, embedding)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        return embedding, inference_time
    
    def batch_optimize_inference(self, image_arrays: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Optimized batch inference for multiple images.
        
        Returns: (embeddings, inference_time_ms)
        """
        start_time = time.time()
        
        # Check cache for each image
        cached_results = []
        uncached_indices = []
        uncached_images = []
        
        for idx, image_array in enumerate(image_arrays):
            cached = self.cache.get(image_array)
            if cached is not None:
                cached_results.append((idx, cached))
            else:
                uncached_indices.append(idx)
                uncached_images.append(image_array)
        
        # Process uncached images in batch
        all_embeddings = [None] * len(image_arrays)
        
        if uncached_images:
            # Stack images
            image_batch = np.stack(uncached_images)
            image_tensor = torch.from_numpy(image_batch).float()
            
            if self.use_gpu:
                image_tensor = image_tensor.cuda()
            
            # Batch inference
            with torch.no_grad():
                embeddings = self.model(image_tensor)
            
            embeddings = embeddings.cpu().numpy()
            
            # Cache and store results
            for i, (original_idx, embedding) in enumerate(zip(uncached_indices, embeddings)):
                if self.quantize_embeddings:
                    embedding = ModelQuantizer.quantize_embedding(embedding, bits=8)
                
                self.cache.put(uncached_images[i], embedding)
                all_embeddings[original_idx] = embedding
        
        # Fill in cached results
        for idx, cached in cached_results:
            all_embeddings[idx] = cached
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        return np.array(all_embeddings), inference_time
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "cache": cache_stats,
            "avg_inference_time_ms": float(np.mean(self.inference_times)) if self.inference_times else 0,
            "min_inference_time_ms": float(np.min(self.inference_times)) if self.inference_times else 0,
            "max_inference_time_ms": float(np.max(self.inference_times)) if self.inference_times else 0,
            "gpu_enabled": self.use_gpu,
            "quantization_enabled": self.quantize_embeddings,
            "total_inferences": len(self.inference_times)
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.inference_times.clear()
        self.cache.clear()


class AdaptiveInferenceScheduler:
    """Adaptively adjust inference parameters based on system load."""
    
    def __init__(self, optimizer: InferenceOptimizer):
        self.optimizer = optimizer
        self.system_load = 0.0
        self.adaptive_batch_size = 8
        self.adaptive_cache_size = 1000
    
    def update_system_load(self, load: float) -> None:
        """Update system load (0.0 to 1.0)."""
        self.system_load = max(0.0, min(1.0, load))
        
        # Adjust batch size based on load
        if self.system_load > 0.8:
            self.adaptive_batch_size = 4  # Reduce batch size under high load
        elif self.system_load < 0.3:
            self.adaptive_batch_size = 16  # Increase batch size under low load
        else:
            self.adaptive_batch_size = 8
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on current load."""
        return self.adaptive_batch_size
    
    def get_optimization_recommendation(self) -> Dict:
        """Get recommendations for optimization."""
        stats = self.optimizer.get_optimization_stats()
        
        recommendations = {
            "cache_hit_rate": stats["cache"]["hit_rate"],
            "avg_inference_time_ms": stats["avg_inference_time_ms"],
            "recommendations": []
        }
        
        # Recommend cache size increase if hit rate is high
        if stats["cache"]["hit_rate"] > 0.8 and stats["cache"]["size"] == stats["cache"]["max_size"]:
            recommendations["recommendations"].append("Consider increasing cache size")
        
        # Recommend batch processing if inference time is high
        if stats["avg_inference_time_ms"] > 50:
            recommendations["recommendations"].append("Use batch processing for better throughput")
        
        # Recommend quantization if GPU memory is constrained
        if not stats["quantization_enabled"] and stats["avg_inference_time_ms"] > 100:
            recommendations["recommendations"].append("Enable quantization to reduce memory usage")
        
        return recommendations
