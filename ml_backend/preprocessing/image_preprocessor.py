"""
Image Preprocessing Pipeline for Sketch-to-Face Recognition

Converts images to grayscale → CLAHE contrast enhancement → resize → normalize to [-1, 1].
Grayscale domain is used for both sketches and photos to minimize the sketch/photo domain gap.
Normalization to [-1, 1] matches the VGGFace2-pretrained InceptionResnetV1 backbone.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class ImagePreprocessor:
    """
    Preprocessing pipeline for sketch-to-face recognition.

    Output: float32 ndarray (H, W) in [-1, 1], suitable for 3-channel replication
    before feeding to InceptionResnetV1 or ResNet50.
    """

    def __init__(self, target_size: Tuple[int, int] = (160, 160), use_face_alignment: bool = False):
        self.target_size = target_size
        self.use_face_alignment = use_face_alignment and MEDIAPIPE_AVAILABLE
        self.face_detector = None

        if self.use_face_alignment:
            try:
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe: {e}")
                self.use_face_alignment = False

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full preprocessing pipeline.

        Returns:
            (preprocessed_image, metadata)
            preprocessed_image: float32 (H, W) in [-1, 1]
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        metadata = {
            'original': image.copy(),
            'original_shape': image.shape,
            'steps': {}
        }

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_face_alignment:
            image_rgb, face_bbox = self._detect_and_align_face(image_rgb)
            metadata['face_bbox'] = face_bbox

        metadata['steps']['after_face_detection'] = image_rgb.copy()

        # Grayscale — common domain for sketch and photo
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        metadata['steps']['after_grayscale'] = image_gray.copy()

        # Resize
        image_resized = cv2.resize(image_gray, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        metadata['steps']['after_resize'] = image_resized.copy()

        # CLAHE: gentler contrast enhancement than global histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_enhanced = clahe.apply(image_resized)
        metadata['steps']['after_clahe'] = image_enhanced.copy()

        # Normalize to [-1, 1] — matches InceptionResnetV1 (VGGFace2) expected range
        image_normalized = (image_enhanced.astype(np.float32) - 127.5) / 128.0

        metadata['steps']['after_normalization'] = image_normalized.copy()
        metadata['statistics'] = {
            'mean': float(np.mean(image_normalized)),
            'std': float(np.std(image_normalized)),
            'min': float(np.min(image_normalized)),
            'max': float(np.max(image_normalized))
        }

        return image_normalized, metadata

    def _detect_and_align_face(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        if not self.use_face_alignment or self.face_detector is None:
            return image_rgb, None

        try:
            results = self.face_detector.process(image_rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_rgb.shape
                padding = 20
                x_min = max(0, int(bbox.xmin * w) - padding)
                y_min = max(0, int(bbox.ymin * h) - padding)
                x_max = min(w, int((bbox.xmin + bbox.width) * w) + padding)
                y_max = min(h, int((bbox.ymin + bbox.height) * h) + padding)
                face_crop = image_rgb[y_min:y_max, x_min:x_max]
                return face_crop, {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
            return image_rgb, None
        except Exception as e:
            print(f"Face detection error: {e}")
            return image_rgb, None

    def preprocess_batch(self, image_paths: list) -> Tuple[np.ndarray, list]:
        batch = []
        metadata_list = []
        for path in image_paths:
            try:
                preprocessed, metadata = self.preprocess(path)
                batch.append(preprocessed)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error preprocessing {path}: {e}")
        return np.stack(batch, axis=0), metadata_list

    @staticmethod
    def apply_preprocessing_steps(image_path: str) -> Dict[str, np.ndarray]:
        preprocessor = ImagePreprocessor()
        _, metadata = preprocessor.preprocess(image_path)
        return metadata['steps']
