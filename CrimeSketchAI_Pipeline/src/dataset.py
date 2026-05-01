"""
dataset.py — CrimeSketch AI Pipeline
Loads the actual paired datasets from:
    C:/Users/User/OneDrive/Desktop/ippr/datasets/organized/

Dataset 1 — CUHK-style (700 matched pairs):
    dataset1/photos/   Original_Face_XXXX.jpg
    dataset1/sketches/ Pencil_Face_XXXX.jpg
    Pairing: 4-digit numeric suffix must match exactly.

Dataset 2 — AR Face Database-style (182 pairs):
    dataset2/photos/   f-005-01.jpg, f-006-01.jpg, m-009-01.jpg, ...
    dataset2/sketches/ 00001.jpg, 00002.jpg, ...
    Pairing: sequential — sorted photos zipped with sorted sketches.

Both datasets are merged, giving ~882 real paired samples.
"""

import os
import re
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default dataset root — points directly at the user's organized folder
DEFAULT_DATASET_ROOT = r"C:\Users\User\OneDrive\Desktop\ippr\datasets\organized"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1 loader — CUHK-style numeric suffix matching
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset1(ds1_dir: str, cats: list) -> List[Dict]:
    """
    Load dataset1: Original_Face_XXXX.jpg ↔ Pencil_Face_XXXX.jpg
    Matched by the 4-digit suffix. Returns list of pair dicts.
    """
    photos_dir  = os.path.join(ds1_dir, "photos")
    sketches_dir = os.path.join(ds1_dir, "sketches")

    if not os.path.isdir(photos_dir) or not os.path.isdir(sketches_dir):
        print(f"[Dataset1] Skipping — missing photos/ or sketches/ in {ds1_dir}")
        return []

    # Build {4-digit-id: full_path} for photos and sketches
    def _extract_id(filename, prefix):
        m = re.search(r'(\d{4})', filename)
        return m.group(1) if m else None

    photo_map = {}
    for f in os.listdir(photos_dir):
        if Path(f).suffix.lower() in IMG_EXTS:
            fid = _extract_id(f, "Original_Face_")
            if fid:
                photo_map[fid] = os.path.join(photos_dir, f)

    sketch_map = {}
    for f in os.listdir(sketches_dir):
        if Path(f).suffix.lower() in IMG_EXTS:
            fid = _extract_id(f, "Pencil_Face_")
            if fid:
                sketch_map[fid] = os.path.join(sketches_dir, f)

    common_ids = sorted(set(photo_map.keys()) & set(sketch_map.keys()))
    print(f"[Dataset1] Photos: {len(photo_map)} | Sketches: {len(sketch_map)} "
          f"| Matched pairs: {len(common_ids)}")

    pairs = []
    for i, fid in enumerate(common_ids):
        p_img = cv2.imread(photo_map[fid])
        s_img = cv2.imread(sketch_map[fid])
        if p_img is None or s_img is None:
            continue
        p_img = cv2.resize(p_img, (160, 160))
        s_img = cv2.resize(s_img, (160, 160))
        pairs.append({
            "subject_id":   f"ds1_{fid}",
            "sketch_path":  sketch_map[fid],
            "photo_path":   photo_map[fid],
            "sketch_array": s_img,
            "photo_array":  p_img,
            "category":     cats[i % len(cats)],
            "source":       "dataset1",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2 loader — AR Face Database-style sequential pairing
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset2(ds2_dir: str, cats: list) -> List[Dict]:
    """
    Load dataset2: f-005-01.jpg, f-006-01.jpg, m-009-01.jpg, ...
                   ↔ 00001.jpg, 00002.jpg, ...
    Sequential pairing: sorted photos[i] ↔ sorted sketches[i]
    """
    photos_dir   = os.path.join(ds2_dir, "photos")
    sketches_dir = os.path.join(ds2_dir, "sketches")

    if not os.path.isdir(photos_dir) or not os.path.isdir(sketches_dir):
        print(f"[Dataset2] Skipping — missing photos/ or sketches/ in {ds2_dir}")
        return []

    # Collect only image files (skip .dat files)
    photos_sorted = sorted([
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    ])
    sketches_sorted = sorted([
        os.path.join(sketches_dir, f)
        for f in os.listdir(sketches_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    ])

    n_pairs = min(len(photos_sorted), len(sketches_sorted))
    print(f"[Dataset2] Photos: {len(photos_sorted)} | Sketches: {len(sketches_sorted)} "
          f"| Sequential pairs: {n_pairs}")

    pairs = []
    for i, (p_path, s_path) in enumerate(
            zip(photos_sorted[:n_pairs], sketches_sorted[:n_pairs])):
        p_img = cv2.imread(p_path)
        s_img = cv2.imread(s_path)
        if p_img is None or s_img is None:
            continue
        p_img = cv2.resize(p_img, (160, 160))
        s_img = cv2.resize(s_img, (160, 160))
        subj_id = f"ds2_{Path(p_path).stem}"
        pairs.append({
            "subject_id":   subj_id,
            "sketch_path":  s_path,
            "photo_path":   p_path,
            "sketch_array": s_img,
            "photo_array":  p_img,
            "category":     cats[i % len(cats)],
            "source":       "dataset2",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main loader — merges both datasets
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES = ["frontal", "semi", "low_qual", "high_det", "composite", "synthetic"]


def load_paired_dataset(
    data_dir: str = DEFAULT_DATASET_ROOT,
    synthetic: bool = False,      # kept for API compat; ignored if real data found
    max_subjects: Optional[int] = None,
    sketch_style: str = "pencil", # ignored for real dataset
    seed: int = 42,
) -> List[Dict]:
    """
    Load all paired sketch-photo samples from the organized dataset folder.

    Expects:
        <data_dir>/dataset1/photos/    Original_Face_XXXX.jpg
        <data_dir>/dataset1/sketches/  Pencil_Face_XXXX.jpg
        <data_dir>/dataset2/photos/    f-005-01.jpg, m-009-01.jpg, ...
        <data_dir>/dataset2/sketches/  00001.jpg, 00002.jpg, ...

    Returns list of dicts:
        {"subject_id", "sketch_path", "photo_path",
         "sketch_array", "photo_array", "category", "source"}
    """
    random.seed(seed)
    np.random.seed(seed)

    cats = CATEGORIES

    # Try structured dataset folder
    ds1_dir = os.path.join(data_dir, "dataset1")
    ds2_dir = os.path.join(data_dir, "dataset2")

    pairs = []

    if os.path.isdir(ds1_dir):
        pairs += _load_dataset1(ds1_dir, cats)
    else:
        print(f"[Dataset] dataset1 not found at {ds1_dir}")

    if os.path.isdir(ds2_dir):
        pairs += _load_dataset2(ds2_dir, cats)
    else:
        print(f"[Dataset] dataset2 not found at {ds2_dir}")

    # Fallback: if the user passed a flat dir with photos/ and sketches/ directly
    if not pairs and os.path.isdir(os.path.join(data_dir, "photos")):
        print("[Dataset] Falling back to flat photos/ + sketches/ structure...")
        pairs += _load_flat(data_dir, cats)

    if not pairs and synthetic:
        print("[Dataset] No real pairs found — generating synthetic sketches...")
        pairs += _synthetic_from_photos(data_dir, cats, sketch_style)

    if not pairs:
        raise FileNotFoundError(
            f"No paired samples found in: {data_dir}\n"
            f"Expected subdirectories: dataset1/ and/or dataset2/\n"
            f"each containing photos/ and sketches/ subfolders."
        )

    # Shuffle deterministically
    random.shuffle(pairs)

    if max_subjects is not None:
        pairs = pairs[:max_subjects]

    print(f"\n[Dataset] ✓ Total paired samples loaded: {len(pairs)}")
    _print_source_breakdown(pairs)
    return pairs


def _print_source_breakdown(pairs):
    from collections import Counter
    sources = Counter(p["source"] for p in pairs)
    cats    = Counter(p["category"] for p in pairs)
    print(f"  Sources  : {dict(sources)}")
    print(f"  Categories: {dict(cats)}")


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: flat photos/ + sketches/ (generic)
# ─────────────────────────────────────────────────────────────────────────────

def _load_flat(data_dir: str, cats: list) -> List[Dict]:
    photos_dir   = os.path.join(data_dir, "photos")
    sketches_dir = os.path.join(data_dir, "sketches")
    if not os.path.isdir(photos_dir):
        return []

    photo_map = {Path(f).stem: os.path.join(photos_dir, f)
                 for f in os.listdir(photos_dir)
                 if Path(f).suffix.lower() in IMG_EXTS}

    if os.path.isdir(sketches_dir):
        sketch_map = {Path(f).stem: os.path.join(sketches_dir, f)
                      for f in os.listdir(sketches_dir)
                      if Path(f).suffix.lower() in IMG_EXTS}
        common = sorted(set(photo_map) & set(sketch_map))
    else:
        # Synthetic fallback
        common = []

    pairs = []
    for i, stem in enumerate(common):
        p = cv2.imread(photo_map[stem])
        s = cv2.imread(sketch_map[stem])
        if p is None or s is None:
            continue
        pairs.append({
            "subject_id":   stem,
            "sketch_path":  sketch_map[stem],
            "photo_path":   photo_map[stem],
            "sketch_array": cv2.resize(s, (160, 160)),
            "photo_array":  cv2.resize(p, (160, 160)),
            "category":     cats[i % len(cats)],
            "source":       "flat",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sketch generator (OpenCV pencil/edge simulation)
# ─────────────────────────────────────────────────────────────────────────────

def photo_to_sketch(img_bgr: np.ndarray, style: str = "pencil") -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if style == "pencil":
        inv  = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
    else:
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blur, 30, 100)
        sketch = cv2.bitwise_not(edges)
    noise  = np.random.normal(0, 5, sketch.shape).astype(np.int16)
    sketch = np.clip(sketch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def _synthetic_from_photos(data_dir: str, cats: list, style: str) -> List[Dict]:
    photos_dir = os.path.join(data_dir, "photos")
    if not os.path.isdir(photos_dir):
        return []
    photo_files = sorted([
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    ])
    pairs = []
    for i, pf in enumerate(photo_files):
        img = cv2.imread(pf)
        if img is None:
            continue
        img = cv2.resize(img, (160, 160))
        sketch = photo_to_sketch(img, style=style)
        pairs.append({
            "subject_id":   Path(pf).stem,
            "sketch_path":  pf,
            "photo_path":   pf,
            "sketch_array": sketch,
            "photo_array":  img,
            "category":     cats[i % len(cats)],
            "source":       "synthetic",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    pairs: List[Dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train   = shuffled[:n_train]
    val     = shuffled[n_train:n_train + n_val]
    test    = shuffled[n_train + n_val:]
    print(f"[Dataset] Split — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch wrappers (unchanged API for models.py)
# ─────────────────────────────────────────────────────────────────────────────

class SketchPhotoDataset(Dataset):
    def __init__(self, pairs: List[Dict], transform=None):
        self.pairs   = pairs
        self.transform = transform
        self.subjects  = sorted({p["subject_id"] for p in pairs})
        self.subj2idx  = {s: i for i, s in enumerate(self.subjects)}

    def __len__(self):
        return len(self.pairs)

    def _to_tensor(self, img_bgr):
        import torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self.transform:
            return self.transform(pil)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return (self._to_tensor(p["sketch_array"]),
                self._to_tensor(p["photo_array"]),
                self.subj2idx[p["subject_id"]])


class ContrastivePairDataset(Dataset):
    def __init__(self, pairs: List[Dict], n_pairs: int = 10000, transform=None):
        self.n_pairs   = n_pairs
        self.transform = transform
        self.by_subject = {}
        for p in pairs:
            self.by_subject.setdefault(p["subject_id"], []).append(p)
        self.subjects = list(self.by_subject.keys())
        self._pregenerate()

    def _pregenerate(self):
        random.seed(42)
        self.generated = []
        half = self.n_pairs // 2
        for _ in range(half):
            subj  = random.choice(self.subjects)
            entry = random.choice(self.by_subject[subj])
            self.generated.append((entry["sketch_array"], entry["photo_array"], 1))
        for _ in range(half):
            s1, s2 = random.sample(self.subjects, 2)
            sketch = random.choice(self.by_subject[s1])["sketch_array"]
            photo  = random.choice(self.by_subject[s2])["photo_array"]
            self.generated.append((sketch, photo, 0))
        random.shuffle(self.generated)

    def __len__(self):
        return len(self.generated)

    def _to_tensor(self, img_bgr):
        import torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        arr = np.array(rgb, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def __getitem__(self, idx):
        import torch
        s, p, label = self.generated[idx]
        return self._to_tensor(s), self._to_tensor(p), torch.tensor(label, dtype=torch.float32)
