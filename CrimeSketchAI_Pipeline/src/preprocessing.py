"""
preprocessing.py — CrimeSketch AI Pipeline
Full preprocessing pipeline:
    1. CLAHE contrast enhancement (clip=2.0, tile=8x8)
    2. MTCNN face detection + 5-keypoint alignment
    3. 160×160 canonical crop + normalisation

Matches exactly the pipeline described in the IEEE FICV 2026 paper.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE Enhancement
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel (LAB colour space).
    Preserves colour information; enhances local contrast without amplifying noise.
    For grayscale/sketch inputs, operates on the single channel directly.
    """
    if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        gray = img_bgr if img_bgr.ndim == 2 else img_bgr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# MTCNN Face Alignment
# ─────────────────────────────────────────────────────────────────────────────

class MTCNNAligner:
    """
    Wraps facenet-pytorch's MTCNN for face detection and similarity-transform
    alignment to a canonical 160×160 crop.
    """

    # Reference 5 landmark positions for a 160×160 output crop
    _REF_LANDMARKS = np.array([
        [30.29459953, 51.69630051],   # left eye
        [65.53179932, 51.50139999],   # right eye
        [48.02519989, 71.73660278],   # nose tip
        [33.54930115, 92.36550140],   # left mouth
        [62.72990036, 92.20410156],   # right mouth
    ], dtype=np.float32)

    def __init__(self, device: str = "cpu"):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device,
                keep_all=False,
            )
            self._available = True
            print("[Preprocessing] MTCNN loaded successfully.")
        except ImportError:
            self.mtcnn = None
            self._available = False
            print("[Preprocessing] WARNING: facenet-pytorch not found. "
                  "Face alignment disabled — using centre crop fallback.")

    def align(self, img_bgr: np.ndarray, target_size: int = 160) -> np.ndarray:
        """
        Detect and align face. Returns aligned BGR image (target_size × target_size).
        Falls back to centre crop if no face is detected.
        """
        if not self._available:
            return self._centre_crop(img_bgr, target_size)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        try:
            import torch
            with torch.no_grad():
                boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or landmarks is None:
                return self._centre_crop(img_bgr, target_size)

            # Use highest-confidence detection
            best_idx = int(np.argmax(probs))
            lm = landmarks[best_idx].astype(np.float32)  # shape (5, 2)

            M = self._similarity_transform(lm, self._REF_LANDMARKS)
            aligned_rgb = cv2.warpAffine(
                img_rgb, M, (target_size, target_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

        except Exception:
            return self._centre_crop(img_bgr, target_size)

    @staticmethod
    def _similarity_transform(src_pts: np.ndarray,
                               dst_pts: np.ndarray) -> np.ndarray:
        """Estimate 2D similarity transform (scale + rotation + translation)."""
        src = src_pts.astype(np.float64)
        dst = dst_pts.astype(np.float64)

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean

        src_std = np.sqrt((src_c ** 2).sum() / len(src))
        dst_std = np.sqrt((dst_c ** 2).sum() / len(dst))

        if src_std < 1e-6:
            return np.eye(2, 3, dtype=np.float32)

        src_norm = src_c / src_std
        dst_norm = dst_c / dst_std

        H = src_norm.T @ dst_norm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        scale = dst_std / src_std * np.trace(np.diag(S) / (src_norm ** 2).sum())
        scale = max(0.5, min(scale, 2.0))

        t = dst_mean - scale * R @ src_mean
        M = np.hstack([scale * R, t.reshape(2, 1)])
        return M.astype(np.float32)

    @staticmethod
    def _centre_crop(img_bgr: np.ndarray, size: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img_bgr[y0:y0+s, x0:x0+s]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Full CrimeSketch Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CrimeSketchPreprocessor:
    """
    End-to-end preprocessing as described in the paper:
        sketch_bgr  →  CLAHE  →  MTCNN align  →  normalised float32 tensor

    Usage:
        pre = CrimeSketchPreprocessor()
        tensor = pre(img_bgr)   # returns torch.Tensor [3, 160, 160]
    """

    def __init__(self, device: str = "cpu",
                 clahe_clip: float = 2.0,
                 clahe_tile: Tuple[int, int] = (8, 8),
                 use_clahe: bool = True,
                 use_align: bool = True):
        self.use_clahe = use_clahe
        self.use_align = use_align
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.aligner = MTCNNAligner(device=device) if use_align else None

    def process_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """Returns preprocessed BGR uint8 image (160×160×3)."""
        out = img_bgr.copy()
        if self.use_clahe:
            out = apply_clahe(out, self.clahe_clip, self.clahe_tile)
        if self.use_align and self.aligner is not None:
            out = self.aligner.align(out, target_size=160)
        else:
            out = cv2.resize(out, (160, 160))
        return out

    def __call__(self, img_bgr: np.ndarray) -> "torch.Tensor":
        """Full pipeline → normalised torch.Tensor [3, 160, 160]."""
        import torch
        processed = self.process_array(img_bgr)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    def batch_process(self, images_bgr):
        """Process a list of BGR images → list of torch.Tensor."""
        return [self(img) for img in images_bgr]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal preprocessor (no CLAHE, no alignment) — for baselines
# ─────────────────────────────────────────────────────────────────────────────

def simple_preprocess(img_bgr: np.ndarray, size: int = 160) -> "torch.Tensor":
    """Resize + normalise only. Used for baseline models."""
    import torch
    img = cv2.resize(img_bgr, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return torch.from_numpy(rgb.transpose(2, 0, 1))
