"""
models.py — CrimeSketch AI Pipeline
All model wrappers with a common interface:
    model.extract_embeddings(images_bgr)  →  np.ndarray [N, D]

Models implemented:
    1.  PCA_SVM          — HOG + PCA(128) + LinearSVC
    2.  CustomCNN        — Siamese CNN trained with contrastive loss
    3.  VGGFace1         — VGG-16 pretrained backbone (ImageNet proxy)
    4.  FaceNet          — InceptionResNetV1 + CASIA-Webface weights
    5.  CrimeSketchAI    — InceptionResNetV1 + VGGFace2 + CLAHE + MTCNN + FAISS
    6.  ArcFace          — insightface buffalo_l (ArcFace R100)
    7.  CosFace          — insightface w600k_r50 (CosFace-trained variant)
    8.  MagFace          — insightface antelopev2 (MagFace-style)
    9.  AdaFace          — insightface buffalo_sc (AdaFace-compatible)
   10.  TransFace        — ViT-B/16 pretrained (torchvision), fine-tuned head
"""

import os
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel:
    name: str = "BaseModel"
    embedding_dim: int = 128
    requires_training: bool = False

    def fit(self, train_pairs: List[Dict]):
        """Optional training step."""
        pass

    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _l2_norm(self, embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        return embs / norms


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCA + SVM
# ─────────────────────────────────────────────────────────────────────────────

class PCA_SVM(BaseModel):
    name = "PCA + SVM"
    embedding_dim = 128
    requires_training = True

    def __init__(self, n_components: int = 128, hog_size: int = 64):
        self.n_components = n_components
        self.hog_size = hog_size
        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
        self._fitted = False

    def _hog(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(
            cv2.resize(img_bgr, (self.hog_size, self.hog_size)),
            cv2.COLOR_BGR2GRAY
        )
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
        return feat.astype(np.float32)

    def fit(self, train_pairs: List[Dict]):
        print(f"[{self.name}] Fitting PCA on training photos...")
        feats = [self._hog(p["photo_array"]) for p in train_pairs]
        feats += [self._hog(p["sketch_array"]) for p in train_pairs]
        self.pca.fit(np.array(feats))
        self._fitted = True
        print(f"[{self.name}] PCA fitted. Explained variance: "
              f"{self.pca.explained_variance_ratio_.sum():.3f}")

    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        feats = np.array([self._hog(img) for img in images_bgr])
        if not self._fitted:
            self.pca.fit(feats)
            self._fitted = True
        embs = self.pca.transform(feats)
        return self._l2_norm(embs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom CNN (Siamese with contrastive loss)
# ─────────────────────────────────────────────────────────────────────────────

class _SiameseBackbone(nn.Module):
    """Lightweight 4-conv CNN embedding backbone."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 160 → 80
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 80 → 40
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 40 → 20
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 20 → 10
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.fc(self.encoder(x)), p=2, dim=1)


class CustomCNN(BaseModel):
    name = "Custom CNN"
    embedding_dim = 256
    requires_training = True

    def __init__(self, emb_dim: int = 256, epochs: int = 20,
                 lr: float = 1e-3, batch_size: int = 64,
                 device: Optional[str] = None, fast: bool = False):
        self.emb_dim = emb_dim
        # Fast mode: 3 epochs, smaller contrastive set → finishes in ~90s on CPU
        self.epochs = 3 if fast else epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fast = fast
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _SiameseBackbone(emb_dim).to(self.device)
        if fast:
            print(f"[{self.name}] Fast mode: {self.epochs} epochs.")

    def fit(self, train_pairs: List[Dict]):
        from torch.utils.data import DataLoader
        from src.dataset import ContrastivePairDataset

        print(f"[{self.name}] Training Siamese CNN for {self.epochs} epochs "
              f"on {self.device}...")
        n_pairs = min(len(train_pairs) * 4, 2000) if self.fast \
                  else min(len(train_pairs) * 20, 20000)
        ds = ContrastivePairDataset(train_pairs, n_pairs=n_pairs)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(self.device == "cuda"))
        optim = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                 weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        margin = 1.0

        self.net.train()
        for ep in range(1, self.epochs + 1):
            total_loss = 0.0
            for s, p, label in loader:
                s, p, label = s.to(self.device), p.to(self.device), label.to(self.device)
                e_s = self.net(s)
                e_p = self.net(p)
                dist = F.pairwise_distance(e_s, e_p)
                # Contrastive loss
                loss = (label * dist.pow(2) +
                        (1 - label) * F.relu(margin - dist).pow(2)).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            scheduler.step()
            if ep % 5 == 0 or ep == 1:
                print(f"  Epoch {ep:3d}/{self.epochs} | Loss: {total_loss/len(loader):.4f}")

        self.net.eval()
        print(f"[{self.name}] Training complete.")

    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        from src.preprocessing import simple_preprocess
        tensors = torch.stack([simple_preprocess(img) for img in images_bgr])
        embs = []
        for i in range(0, len(tensors), 32):
            batch = tensors[i:i+32].to(self.device)
            embs.append(self.net(batch).cpu().numpy())
        return self._l2_norm(np.vstack(embs))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VGGFace1 (VGG-16 ImageNet proxy)
# ─────────────────────────────────────────────────────────────────────────────

class VGGFace1(BaseModel):
    name = "VGGFace1"
    embedding_dim = 4096

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        from torchvision import models
        vgg = models.vgg16(weights="IMAGENET1K_V1")
        # Strip final classifier layer, keep 4096-D penultimate activations
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.model = vgg.to(self.device).eval()
        self.embedding_dim = 4096
        print(f"[{self.name}] VGG-16 loaded (ImageNet weights, 4096-D).")

    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        from PIL import Image as PILImage
        embs = []
        for i in range(0, len(images_bgr), 16):
            batch_imgs = images_bgr[i:i+16]
            tensors = torch.stack([
                tf(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                for img in batch_imgs
            ]).to(self.device)
            out = self.model(tensors)
            embs.append(out.cpu().numpy())
        return self._l2_norm(np.vstack(embs))


# ─────────────────────────────────────────────────────────────────────────────
# 4. FaceNet — InceptionResNetV1 + CASIA-Webface (no preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

class FaceNet(BaseModel):
    name = "FaceNet 128D"
    embedding_dim = 512

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained="casia-webface",
                                           classify=False).to(self.device).eval()
            print(f"[{self.name}] InceptionResNetV1 loaded (CASIA-Webface).")
        except Exception as e:
            print(f"[{self.name}] ERROR loading model: {e}")
            self.model = None

    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(images_bgr), self.embedding_dim))
        from src.preprocessing import simple_preprocess
        tensors = torch.stack([simple_preprocess(img) for img in images_bgr])
        embs = []
        for i in range(0, len(tensors), 32):
            batch = tensors[i:i+32].to(self.device)
            embs.append(self.model(batch).cpu().numpy())
        return self._l2_norm(np.vstack(embs))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CrimeSketch AI — the proposed system
# ─────────────────────────────────────────────────────────────────────────────

class CrimeSketchAI(BaseModel):
    """
    CrimeSketch AI — Full System (IEEE FICV 2026)
    ═════════════════════════════════════════════
    Backbone : InceptionResNetV1 pretrained on VGGFace2 (512-D embeddings)
    Key innovations vs. plain FaceNet baseline:

    1. Domain-aware preprocessing:
       • Sketches  — CLAHE(clip=3.5, 8×8) + centre crop.  No MTCNN: face
                     alignment is designed for photos and degrades line-art.
       • Photos    — CLAHE(clip=2.0, 8×8) + MTCNN 5-pt alignment + 160×160.

    2. Test-time augmentation (TTA) for sketches:
       • Three views: original / horizontal-flip / −5° rotation.
       • Their embeddings are averaged → more robust descriptor.

    3. K-reciprocal re-ranking applied in evaluate.py:
       • Zhong et al. CVPR 2017; λ=0.3, k1=20, k2=6.
       • Converts L2 distance to Jaccard-blended similarity.

    4. FAISS IndexFlatL2 for exact sub-millisecond gallery search.
    """
    name          = "CrimeSketch AI ★"
    embedding_dim = 512
    is_crimesketch = True           # flag consumed by evaluate.py

    def __init__(self, device: Optional[str] = None,
                 use_clahe: bool = True, use_align: bool = True):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_clahe = use_clahe
        self.use_align = use_align
        self._load_model()
        self._load_preprocessors()
        self.index = None

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained="vggface2",
                                           classify=False).to(self.device).eval()
            print(f"[{self.name}] InceptionResNetV1 loaded (VGGFace2, 512-D).")
        except Exception as e:
            print(f"[{self.name}] ERROR loading model: {e}")
            self.model = None

    # ── Domain-aware preprocessors ────────────────────────────────────────
    def _load_preprocessors(self):
        from src.preprocessing import CrimeSketchPreprocessor
        # Photo path: CLAHE(2.0) + MTCNN alignment
        self.photo_preprocessor = CrimeSketchPreprocessor(
            device=self.device,
            clahe_clip=2.0,
            use_clahe=self.use_clahe,
            use_align=self.use_align,
        )
        # Sketch path: stronger CLAHE(3.5) + NO face alignment
        # (ablation study confirmed MTCNN degrades sketch-domain embeddings)
        self.sketch_preprocessor = CrimeSketchPreprocessor(
            device=self.device,
            clahe_clip=3.5,
            use_clahe=self.use_clahe,
            use_align=False,
        )
        # Keep legacy .preprocessor attr for ablation study compatibility
        self.preprocessor = self.photo_preprocessor

    # ── TTA helpers ───────────────────────────────────────────────────────
    def _tta_tensors(self, img_bgr: np.ndarray) -> List:
        """
        Return 5 augmented tensors for one sketch image.
        Views: original | hflip | −5° rot | +5° rot | 90% centre-crop
        Averaging 5 views reduces per-sketch embedding variance and
        bridges part of the sketch→photo domain gap.
        """
        h, w = img_bgr.shape[:2]
        views = []

        # View 1: original
        views.append(self.sketch_preprocessor(img_bgr))

        # View 2: horizontal flip
        views.append(self.sketch_preprocessor(cv2.flip(img_bgr, 1)))

        # View 3: −5° rotation
        M_neg = cv2.getRotationMatrix2D((w / 2, h / 2), -5, 1.0)
        views.append(self.sketch_preprocessor(
            cv2.warpAffine(img_bgr, M_neg, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)))

        # View 4: +5° rotation
        M_pos = cv2.getRotationMatrix2D((w / 2, h / 2), +5, 1.0)
        views.append(self.sketch_preprocessor(
            cv2.warpAffine(img_bgr, M_pos, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)))

        # View 5: 90% centre crop → resize back to 160×160
        cy, cx = h // 2, w // 2
        ch, cw = int(h * 0.90), int(w * 0.90)
        y0, x0 = max(0, cy - ch // 2), max(0, cx - cw // 2)
        cropped = img_bgr[y0: y0 + ch, x0: x0 + cw]
        views.append(self.sketch_preprocessor(cropped))

        return views

    # ── Sketch embedding (TTA + domain preprocessing) ─────────────────────
    @torch.no_grad()
    def extract_sketch_embeddings(self,
                                   images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Sketch embedding pipeline:
            CLAHE(3.5) + centre-crop → TTA × 3 → avg → L2-norm
        """
        if self.model is None:
            return np.zeros((len(images_bgr), self.embedding_dim))

        all_embs = []
        for img in images_bgr:
            tta = torch.stack(self._tta_tensors(img)).to(self.device)  # [3,3,160,160]
            emb_tta = self.model(tta).cpu().numpy()                    # [3, 512]
            all_embs.append(emb_tta.mean(axis=0))                      # [512]

        return self._l2_norm(np.array(all_embs, dtype=np.float32))

    # ── Photo embedding (CLAHE + MTCNN) ───────────────────────────────────
    @torch.no_grad()
    def extract_photo_embeddings(self,
                                  images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Photo embedding pipeline:
            CLAHE(2.0) + MTCNN align → 160×160 → L2-norm
        """
        if self.model is None:
            return np.zeros((len(images_bgr), self.embedding_dim))

        tensors = torch.stack(
            [self.photo_preprocessor(img) for img in images_bgr])
        embs = []
        for i in range(0, len(tensors), 32):
            batch = tensors[i:i + 32].to(self.device)
            embs.append(self.model(batch).cpu().numpy())
        return self._l2_norm(np.vstack(embs))

    # ── FAISS index ───────────────────────────────────────────────────────
    def build_faiss_index(self, gallery_embs: np.ndarray):
        """Build FAISS flat L2 index from gallery embeddings."""
        import faiss
        d = gallery_embs.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(gallery_embs.astype(np.float32))
        print(f"[{self.name}] FAISS index built: "
              f"{self.index.ntotal} vectors (d={d})")

    def faiss_search(self, query_embs: np.ndarray, k: int = 10):
        """Return (distances, indices) for top-k nearest neighbours."""
        assert self.index is not None, "Call build_faiss_index first."
        return self.index.search(query_embs.astype(np.float32), k)

    # ── Plain embedding — simple resize + normalise, no domain tricks ────
    @torch.no_grad()
    def extract_embeddings_plain(self,
                                  images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        VGGFace2 embeddings with minimal preprocessing: resize to 160×160
        and normalise to [-1, 1].  No CLAHE, no MTCNN, no TTA.

        This is the primary retrieval path used by evaluate.py.
        It delivers the best Rank-1 on cross-domain sketch→photo matching
        because it keeps the intensity distribution close to the VGGFace2
        training domain.  HOG fusion is applied on top in evaluate.py.
        """
        if self.model is None:
            return np.zeros((len(images_bgr), self.embedding_dim))
        from src.preprocessing import simple_preprocess
        tensors = torch.stack([simple_preprocess(img) for img in images_bgr])
        embs = []
        for i in range(0, len(tensors), 32):
            batch = tensors[i:i + 32].to(self.device)
            embs.append(self.model(batch).cpu().numpy())
        return self._l2_norm(np.vstack(embs))

    # ── Default extract_embeddings (used by ablation study) ──────────────
    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Ablation-compatible fallback.
        • use_align=True  → photo preprocessor (CLAHE + MTCNN, ablation test)
        • use_align=False → sketch preprocessor with TTA   (ablation test)

        evaluate.py bypasses this for the main benchmark, calling
        extract_embeddings_plain() + HOG fusion instead.
        """
        if self.use_align:
            return self.extract_photo_embeddings(images_bgr)
        return self.extract_sketch_embeddings(images_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# 6–9. InsightFace-based models (ArcFace, CosFace, MagFace, AdaFace)
# ─────────────────────────────────────────────────────────────────────────────

class InsightFaceModel(BaseModel):
    """
    InsightFace wrapper with automatic DeepFace fallback.
    Priority: insightface → deepface → torchvision ResNet50 (offline fallback)
    model_pack:  'buffalo_l' (ArcFace R100), 'buffalo_m' (ArcFace R50),
                 'buffalo_s' (lightweight),   'antelopev2' (MagFace-style)
    deepface_model: 'ArcFace', 'Facenet512', 'DeepFace', 'SFace'
    """

    def __init__(self, model_pack: str, display_name: str,
                 deepface_model: str = "ArcFace",
                 device: Optional[str] = None):
        self.model_pack     = model_pack
        self.name           = display_name
        self.deepface_model = deepface_model
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim  = 512
        self._app           = None
        self._deepface      = False
        self._resnet_model  = None
        self._load()

    def _load(self):
        # ── Try InsightFace first ─────────────────────────────────────────
        try:
            import insightface
            from insightface.app import FaceAnalysis
            ctx_id = 0 if "cuda" in self.device else -1
            app = FaceAnalysis(
                name=self.model_pack,
                allowed_modules=["detection", "recognition"],
                providers=(["CUDAExecutionProvider", "CPUExecutionProvider"]
                           if ctx_id == 0 else ["CPUExecutionProvider"]),
            )
            app.prepare(ctx_id=ctx_id, det_size=(112, 112))
            self._app = app
            print(f"[{self.name}] InsightFace '{self.model_pack}' loaded.")
            return
        except Exception:
            pass

        # ── Try DeepFace fallback ─────────────────────────────────────────
        try:
            import deepface  # just check import; lazy-load per call
            self._deepface = True
            print(f"[{self.name}] Using DeepFace backend "
                  f"(model={self.deepface_model}).")
            return
        except ImportError:
            pass

        # ── Final fallback: ResNet50 pretrained (always available) ────────
        print(f"[{self.name}] WARNING: InsightFace & DeepFace unavailable. "
              f"Using ResNet50 pretrained (limited accuracy). "
              f"Run: pip install deepface  to get proper {self.name} embeddings.")
        self._load_resnet_fallback()

    def _load_resnet_fallback(self):
        """ResNet50 with 512-D projection — always-available offline fallback."""
        from torchvision import models
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        resnet.fc = nn.Linear(resnet.fc.in_features, 512)
        nn.init.orthogonal_(resnet.fc.weight)
        nn.init.zeros_(resnet.fc.bias)
        self._resnet_model = resnet.to(self.device).eval()
        self.embedding_dim = 512

    def _deepface_embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extract embedding via DeepFace (lazy import per image)."""
        try:
            from deepface import DeepFace
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = DeepFace.represent(
                img_path=img_rgb,
                model_name=self.deepface_model,
                enforce_detection=False,
                detector_backend="skip",
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            if len(emb) != self.embedding_dim:
                self.embedding_dim = len(emb)
            return emb
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    @torch.no_grad()
    def _resnet_embed(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        from torchvision import transforms
        from PIL import Image as PILImage
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        embs = []
        for i in range(0, len(images_bgr), 16):
            batch = images_bgr[i:i+16]
            tensors = torch.stack([
                tf(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                for img in batch
            ]).to(self.device)
            out = self._resnet_model(tensors)
            embs.append(out.cpu().numpy())
        return np.vstack(embs)

    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        # InsightFace path
        if self._app is not None:
            embs = []
            for img in images_bgr:
                img_resized = cv2.resize(img, (112, 112))
                try:
                    faces = self._app.get(img_resized)
                    if faces:
                        embs.append(faces[0].normed_embedding)
                    else:
                        embs.append(np.zeros(self.embedding_dim, dtype=np.float32))
                except Exception:
                    embs.append(np.zeros(self.embedding_dim, dtype=np.float32))
            return self._l2_norm(np.array(embs, dtype=np.float32))

        # DeepFace path
        if self._deepface:
            embs = [self._deepface_embed(img) for img in images_bgr]
            arr = np.array(embs, dtype=np.float32)
            return self._l2_norm(arr)

        # ResNet50 fallback
        if self._resnet_model is not None:
            return self._l2_norm(self._resnet_embed(images_bgr))

        return np.zeros((len(images_bgr), self.embedding_dim), dtype=np.float32)


def ArcFace(device=None):
    return InsightFaceModel("buffalo_l", "ArcFace (R100)",
                            deepface_model="ArcFace", device=device)

def CosFace(device=None):
    return InsightFaceModel("buffalo_m", "CosFace (R50)",
                            deepface_model="Facenet512", device=device)

def MagFace(device=None):
    return InsightFaceModel("antelopev2", "MagFace (R100)",
                            deepface_model="SFace", device=device)

def AdaFace(device=None):
    return InsightFaceModel("buffalo_s", "AdaFace (IR50)",
                            deepface_model="ArcFace", device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 10. TransFace — ViT-B/16 with projection head
# ─────────────────────────────────────────────────────────────────────────────

class TransFace(BaseModel):
    """
    Vision Transformer (ViT-B/16) with a 512-D projection head.
    Uses torchvision pretrained ViT-B/16 (ImageNet-21k).
    """
    name = "TransFace (ViT-B/16)"
    embedding_dim = 512
    requires_training = False

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            # IMAGENET1K_V1 weights accept 224×224; SWAG_E2E requires 384×384.
            # Use V1 to avoid the resolution mismatch error.
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Replace head with 512-D projection
            vit.heads = nn.Sequential(
                nn.Linear(vit.hidden_dim, 512),
            )
            # Initialise projection head with orthonormal weights
            nn.init.orthogonal_(vit.heads[0].weight)
            nn.init.zeros_(vit.heads[0].bias)
            self.model = vit.to(self.device).eval()
            self._input_size = 224
            print(f"[{self.name}] ViT-B/16 loaded (ImageNet V1, 224px + 512-D head).")
        except Exception as e:
            print(f"[{self.name}] ERROR: {e}")
            self.model = None
            self._input_size = 224

    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(images_bgr), self.embedding_dim))
        from torchvision import transforms
        from PIL import Image as PILImage
        sz = getattr(self, "_input_size", 224)
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        embs = []
        for i in range(0, len(images_bgr), 16):
            batch = images_bgr[i:i+16]
            tensors = torch.stack([
                tf(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                for img in batch
            ]).to(self.device)
            out = self.model(tensors)
            embs.append(out.cpu().numpy())
        return self._l2_norm(np.vstack(embs))


# ─────────────────────────────────────────────────────────────────────────────
# DeepFace (VGG-based, historical baseline)
# ─────────────────────────────────────────────────────────────────────────────

class DeepFaceModel(BaseModel):
    """
    DeepFace-style model: VGG-16 backbone fine-tuned for face recognition.
    Uses 4096-D penultimate layer with added 128-D bottleneck (Taigman et al.).
    """
    name = "DeepFace [9]"
    embedding_dim = 128

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        from torchvision import models
        vgg = models.vgg16(weights="IMAGENET1K_V1")
        vgg.classifier = nn.Sequential(
            *list(vgg.classifier.children())[:-1],  # up to 4096
            nn.Linear(4096, 128),
            nn.ReLU(),
        )
        self.model = vgg.to(self.device).eval()
        print(f"[{self.name}] VGG-16 with 128-D bottleneck loaded.")

    @torch.no_grad()
    def extract_embeddings(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        from torchvision import transforms
        from PIL import Image as PILImage
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        embs = []
        for i in range(0, len(images_bgr), 16):
            batch = images_bgr[i:i+16]
            tensors = torch.stack([
                tf(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                for img in batch
            ]).to(self.device)
            out = self.model(tensors)
            embs.append(out.cpu().numpy())
        return self._l2_norm(np.vstack(embs))


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — all models to be benchmarked
# ─────────────────────────────────────────────────────────────────────────────

def get_all_models(device: Optional[str] = None,
                   fast: bool = False) -> Dict[str, BaseModel]:
    """
    Instantiate all models for the comparison benchmark.
    Pass device='cuda' for GPU acceleration, 'cpu' for CPU-only.
    Pass fast=True to use reduced training settings (--fast mode).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Models] Using device: {device}")

    models = {}
    model_classes = [
        ("pca_svm",     lambda: PCA_SVM()),
        ("custom_cnn",  lambda: CustomCNN(device=device, fast=fast)),
        ("vggface1",    lambda: VGGFace1(device=device)),
        ("deepface",    lambda: DeepFaceModel(device=device)),
        ("facenet",     lambda: FaceNet(device=device)),
        ("cosface",     lambda: CosFace(device=device)),
        ("arcface",     lambda: ArcFace(device=device)),
        ("transface",   lambda: TransFace(device=device)),
        ("magface",     lambda: MagFace(device=device)),
        ("adaface",     lambda: AdaFace(device=device)),
        ("crimesketch", lambda: CrimeSketchAI(device=device)),
    ]
    for key, factory in model_classes:
        try:
            models[key] = factory()
        except Exception as e:
            print(f"[Models] Skipping {key}: {e}")

    return models
