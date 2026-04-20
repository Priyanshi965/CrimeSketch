"""
Dataset Loading Utilities for CrimeSketch AI

Handles loading, organizing, and preprocessing datasets for training and indexing.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2


class DatasetLoader:
    """
    Loads and manages datasets for sketch-to-face recognition.
    
    Supports:
    - Dataset 1: Pencil sketches + RGB faces
    - Dataset 2: Cropped/original sketches + photos
    - Dataset 3: CelebA celebrity faces
    """
    
    def __init__(self, base_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "organized")):
        """
        Initialize dataset loader.
        
        Args:
            base_path: Base path to organized datasets
        """
        self.base_path = base_path
        self.datasets = {}
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load information about available datasets."""
        dataset_paths = {
            'dataset1': {
                'sketches': os.path.join(self.base_path, 'dataset1/sketches'),
                'photos': os.path.join(self.base_path, 'dataset1/photos')
            },
            'dataset2': {
                'sketches': os.path.join(self.base_path, 'dataset2/sketches'),
                'photos': os.path.join(self.base_path, 'dataset2/photos')
            },
            'dataset3': {
                'celebrity_faces': os.path.join(self.base_path, 'dataset3/celebrity_faces')
            }
        }
        
        for dataset_name, paths in dataset_paths.items():
            self.datasets[dataset_name] = {
                'paths': paths,
                'sketches': [],
                'photos': [],
                'celebrity_faces': []
            }
            
            # Load file lists
            if 'sketches' in paths and os.path.exists(paths['sketches']):
                self.datasets[dataset_name]['sketches'] = self._get_image_files(paths['sketches'])
            
            if 'photos' in paths and os.path.exists(paths['photos']):
                self.datasets[dataset_name]['photos'] = self._get_image_files(paths['photos'])
            
            if 'celebrity_faces' in paths and os.path.exists(paths['celebrity_faces']):
                self.datasets[dataset_name]['celebrity_faces'] = self._get_image_files(paths['celebrity_faces'])
    
    @staticmethod
    def _get_image_files(directory: str) -> List[str]:
        """
        Get all image files from a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            List of image file paths
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(directory):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(directory, file))
        
        return sorted(image_files)
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'dataset1_sketches': len(self.datasets['dataset1']['sketches']),
            'dataset1_photos': len(self.datasets['dataset1']['photos']),
            'dataset2_sketches': len(self.datasets['dataset2']['sketches']),
            'dataset2_photos': len(self.datasets['dataset2']['photos']),
            'dataset3_celebrity_faces': len(self.datasets['dataset3']['celebrity_faces']),
            'total_sketches': (len(self.datasets['dataset1']['sketches']) + 
                             len(self.datasets['dataset2']['sketches'])),
            'total_photos': (len(self.datasets['dataset1']['photos']) + 
                           len(self.datasets['dataset2']['photos']) +
                           len(self.datasets['dataset3']['celebrity_faces']))
        }
        return stats
    
    def get_paired_data(self) -> Tuple[List[str], List[str]]:
        """
        Get paired sketch-photo data for training.
        
        Returns:
            Tuple of (sketch_paths, photo_paths) with matching indices
        """
        sketches = []
        photos = []
        
        # Dataset 1: Match by index (Pencil_Face_XXXX.jpg -> Original_Face_XXXX.jpg)
        ds1_sketches = self.datasets['dataset1']['sketches']
        ds1_photos = self.datasets['dataset1']['photos']
        
        for sketch_path in ds1_sketches:
            sketch_name = os.path.basename(sketch_path)
            # Extract number from Pencil_Face_XXXX.jpg
            sketch_num = sketch_name.replace('Pencil_Face_', '').replace('.jpg', '')
            
            # Find corresponding photo
            photo_name = f'Original_Face_{sketch_num}.jpg'
            photo_path = os.path.join(os.path.dirname(ds1_photos[0]), photo_name)
            
            if os.path.exists(photo_path):
                sketches.append(sketch_path)
                photos.append(photo_path)
        
        # Dataset 2: Match by index (XXXXX.jpg -> f-XXXXX-01.jpg)
        ds2_sketches = self.datasets['dataset2']['sketches']
        ds2_photos = self.datasets['dataset2']['photos']
        
        for sketch_path in ds2_sketches:
            sketch_name = os.path.basename(sketch_path)
            sketch_num = sketch_name.replace('.jpg', '')
            
            # Find corresponding photo
            photo_name = f'f-{sketch_num}-01.jpg'
            photo_path = os.path.join(os.path.dirname(ds2_photos[0]), photo_name)
            
            if os.path.exists(photo_path):
                sketches.append(sketch_path)
                photos.append(photo_path)
        
        return sketches, photos
    
    def get_all_photos(self) -> List[str]:
        """
        Get all photo paths for indexing.
        
        Returns:
            List of photo paths
        """
        photos = []
        photos.extend(self.datasets['dataset1']['photos'])
        photos.extend(self.datasets['dataset2']['photos'])
        photos.extend(self.datasets['dataset3']['celebrity_faces'])
        return photos
    
    def get_all_sketches(self) -> List[str]:
        """
        Get all sketch paths.
        
        Returns:
            List of sketch paths
        """
        sketches = []
        sketches.extend(self.datasets['dataset1']['sketches'])
        sketches.extend(self.datasets['dataset2']['sketches'])
        return sketches
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image as numpy array.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_train_val_test_split(self, train_ratio: float = 0.7, 
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Create train/validation/test splits ensuring no identity leakage.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with splits: {'train': (sketches, photos), 'val': ..., 'test': ...}
        """
        np.random.seed(random_seed)
        
        sketches, photos = self.get_paired_data()
        n_pairs = len(sketches)
        
        # Create indices
        indices = np.arange(n_pairs)
        np.random.shuffle(indices)
        
        # Split indices
        train_end = int(n_pairs * train_ratio)
        val_end = train_end + int(n_pairs * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        splits = {
            'train': (
                [sketches[i] for i in train_idx],
                [photos[i] for i in train_idx]
            ),
            'val': (
                [sketches[i] for i in val_idx],
                [photos[i] for i in val_idx]
            ),
            'test': (
                [sketches[i] for i in test_idx],
                [photos[i] for i in test_idx]
            )
        }
        
        return splits
    
    def save_dataset_metadata(self, output_path: str):
        """
        Save dataset metadata to JSON file.
        
        Args:
            output_path: Path to save metadata JSON
        """
        metadata = {
            'datasets': self.datasets,
            'stats': self.get_dataset_stats(),
            'paired_data_count': len(self.get_paired_data()[0])
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Test dataset loader
    loader = DatasetLoader()
    
    print("Dataset Statistics:")
    stats = loader.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nPaired data count:", len(loader.get_paired_data()[0]))
    
    print("\nTrain/Val/Test splits:")
    splits = loader.create_train_val_test_split()
    for split_name, (sketches, photos) in splits.items():
        print(f"  {split_name}: {len(sketches)} pairs")
    
    # Save metadata
    loader.save_dataset_metadata(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_metadata.json"))
    print("\nMetadata saved!")
