"""
End-to-end training and evaluation pipeline for sketch-to-face matching.

This module provides:
- Siamese/contrastive training on paired sketch-photo data
- Retrieval evaluation (Rank-1 / Top-K)
- Pairwise metrics (accuracy, precision, recall, F1, ROC-AUC)
- Target gate checking (e.g. >= 94% accuracy)
- Optional reindex after training
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import cv2

from preprocessing.image_preprocessor import ImagePreprocessor
from models.siamese_model import ModelManager, ContrastiveLoss, TripletLoss
from embeddings.faiss_indexer import FAISSIndexer
from database.schema import DatabaseManager
from utils.dataset_loader import DatasetLoader


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 5e-5
    loss_type: str = "triplet"
    target_accuracy: float = 0.94
    top_k: int = 5
    seed: int = 42
    reindex_after_train: bool = True
    max_train_samples: int = 0  # 0 = full dataset
    max_eval_samples: int = 0   # 0 = full dataset
    gate_metric: str = "accuracy"  # accuracy or rank1_accuracy
    resume_from_checkpoint: bool = True
    triplet_margin: float = 0.3   # tighter margin — backbone embeddings are already good
    use_augmentation: bool = True
    batch_hard_triplet: bool = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TripletDataset(Dataset):
    def __init__(self, sketches: List[str], photos: List[str], preprocessor: ImagePreprocessor, augment: bool = False):
        self.sketches = sketches
        self.photos = photos
        self.preprocessor = preprocessor
        self.n = len(sketches)
        self.augment = augment

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Light geometric/photometric augmentation on normalized grayscale image in [-1, 1]."""
        # Decode from [-1, 1] → uint8 [0, 255]
        img = np.clip((image * 128.0 + 127.5), 0, 255).astype(np.uint8)
        h, w = img.shape

        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Random rotation + translation
        angle = random.uniform(-10.0, 10.0)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        mat[:, 2] += [tx, ty]
        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

        # Random contrast / brightness
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Mild Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5.0, size=img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Re-encode to [-1, 1]
        return (img.astype(np.float32) - 127.5) / 128.0

    def __len__(self) -> int:
        return self.n

    def _to_tensor(self, image_path: str) -> torch.Tensor:
        image, _ = self.preprocessor.preprocess(image_path)
        if self.augment:
            image = self._augment(image)
        # (H, W) → (3, H, W): replicate grayscale across 3 channels for backbone
        tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float()
        return tensor

    def __getitem__(self, idx: int):
        anchor = self._to_tensor(self.sketches[idx])
        positive = self._to_tensor(self.photos[idx])
        neg_idx = random.randrange(self.n)
        while neg_idx == idx:
            neg_idx = random.randrange(self.n)
        negative = self._to_tensor(self.photos[neg_idx])

        return anchor, positive, negative


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _to_input_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert preprocessed (H, W) grayscale image to (1, 3, H, W) model input."""
    return torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().to(device)


def _distance(model_manager: ModelManager, sketch_path: str, photo_path: str, preprocessor: ImagePreprocessor, device: torch.device) -> float:
    sketch, _ = preprocessor.preprocess(sketch_path)
    photo, _ = preprocessor.preprocess(photo_path)

    s = _to_input_tensor(sketch, device)
    p = _to_input_tensor(photo, device)

    with torch.no_grad():
        emb_s = model_manager.model.get_embedding(s)
        emb_p = model_manager.model.get_embedding(p)
        dist = torch.norm(emb_s - emb_p, p=2, dim=1)
    return float(dist.item())


def _find_best_threshold(model_manager: ModelManager, sketches: List[str], photos: List[str], preprocessor: ImagePreprocessor, device: torch.device) -> Tuple[float, Dict[str, float], float]:
    y_true: List[int] = []
    y_score: List[float] = []

    n = len(sketches)
    for i in range(n):
        pos_dist = _distance(model_manager, sketches[i], photos[i], preprocessor, device)
        y_true.append(1)
        y_score.append(-pos_dist)

        j = random.randrange(n)
        while j == i:
            j = random.randrange(n)
        neg_dist = _distance(model_manager, sketches[i], photos[j], preprocessor, device)
        y_true.append(0)
        y_score.append(-neg_dist)

    y_true_np = np.array(y_true, dtype=np.int32)
    y_score_np = np.array(y_score, dtype=np.float32)

    candidates = np.quantile(y_score_np, np.linspace(0.05, 0.95, 37))
    best_thr = float(candidates[0])
    best_metrics = {"f1": -1.0}

    for thr in candidates:
        y_pred = (y_score_np >= thr).astype(np.int32)
        metrics = _classification_metrics(y_true_np, y_pred)
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_thr = float(thr)

    auc = _roc_auc_score(y_true_np, y_score_np)
    return best_thr, best_metrics, auc


def _retrieval_metrics(model_manager: ModelManager, sketches: List[str], photos: List[str], preprocessor: ImagePreprocessor, device: torch.device, top_k: int = 5) -> Dict[str, float]:
    gallery_embeddings = []
    for p in photos:
        img, _ = preprocessor.preprocess(p)
        t = _to_input_tensor(img, device)
        with torch.no_grad():
            emb = model_manager.model.get_embedding(t).cpu().numpy()[0]
        gallery_embeddings.append(emb)

    gallery = np.stack(gallery_embeddings, axis=0)
    rank1 = 0
    topk = 0

    for i, s_path in enumerate(sketches):
        s_img, _ = preprocessor.preprocess(s_path)
        s_t = _to_input_tensor(s_img, device)
        with torch.no_grad():
            s_emb = model_manager.model.get_embedding(s_t).cpu().numpy()[0]

        dists = np.linalg.norm(gallery - s_emb[None, :], axis=1)
        ranked = np.argsort(dists)

        if int(ranked[0]) == i:
            rank1 += 1
        if i in ranked[:top_k]:
            topk += 1

    n = max(1, len(sketches))
    return {
        "rank1_accuracy": float(rank1 / n),
        "topk_accuracy": float(topk / n),
    }


def _reindex_with_trained_model(model_manager: ModelManager, preprocessor: ImagePreprocessor, db_path: str, index_path: str, metadata_path: str):
    db = DatabaseManager(db_path)
    indexer = FAISSIndexer(embedding_dim=512, index_type="flat")

    suspects = db.get_suspects_filtered(limit=1_000_000)
    embeddings = []
    suspect_ids = []

    for suspect in suspects:
        image_path = suspect.get("image_path")
        if not image_path or not os.path.exists(image_path):
            continue

        img, _ = preprocessor.preprocess(image_path)
        t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
        emb = model_manager.get_embedding(t)[0]

        db.add_embedding(int(suspect["id"]), emb)
        embeddings.append(emb)
        suspect_ids.append(int(suspect["id"]))

    if embeddings:
        indexer.add_embeddings(np.array(embeddings, dtype=np.float32), suspect_ids)
        indexer.save_index(index_path, metadata_path)

    db.close()


def run_training_pipeline(project_root: str, config: Dict) -> Dict:
    cfg = TrainingConfig(**config)
    _set_seed(cfg.seed)

    ml_backend = Path(project_root) / "ml_backend"
    db_path = str(ml_backend / "database" / "crimesketch.db")
    model_path = str(ml_backend / "models" / "siamese_checkpoint.pt")
    report_path = str(ml_backend / "models" / "training_report.json")
    index_path = str(ml_backend / "embeddings" / "index.faiss")
    metadata_path = str(ml_backend / "embeddings" / "metadata.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = ImagePreprocessor(target_size=(160, 160), use_face_alignment=False)
    loader = DatasetLoader()
    splits = loader.create_train_val_test_split(random_seed=cfg.seed)

    train_sketches, train_photos = splits["train"]
    val_sketches, val_photos = splits["val"]
    test_sketches, test_photos = splits["test"]

    if cfg.max_train_samples > 0:
        train_sketches = train_sketches[: cfg.max_train_samples]
        train_photos = train_photos[: cfg.max_train_samples]
    if cfg.max_eval_samples > 0:
        val_sketches = val_sketches[: cfg.max_eval_samples]
        val_photos = val_photos[: cfg.max_eval_samples]
        test_sketches = test_sketches[: cfg.max_eval_samples]
        test_photos = test_photos[: cfg.max_eval_samples]

    print(
        f"[train] device={device} "
        f"train_pairs={len(train_sketches)} val_pairs={len(val_sketches)} test_pairs={len(test_sketches)}",
        flush=True,
    )

    model_manager = ModelManager(
        model_type="siamese",
        embedding_dim=512,
        device=str(device),
        model_path=None,
    )
    model = model_manager.model
    model.train()

    if cfg.loss_type == "triplet":
        criterion = TripletLoss(margin=cfg.triplet_margin)
    else:
        criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    train_losses: List[float] = []
    start_epoch = 0

    # Resume from last checkpoint if requested
    if cfg.resume_from_checkpoint and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))
                if start_epoch > 0:
                    print(f"[resume] continuing from epoch {start_epoch}/{cfg.epochs}", flush=True)
        except Exception as e:
            print(f"[resume] checkpoint load failed, starting fresh: {e}", flush=True)

    dataset = TripletDataset(train_sketches, train_photos, preprocessor, augment=cfg.use_augmentation)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs - start_epoch),
        eta_min=cfg.learning_rate * 0.1,
    )

    start = time.time()
    for epoch_idx in range(start_epoch, cfg.epochs):
        running = 0.0
        count = 0

        print(f"[train] epoch {epoch_idx + 1}/{cfg.epochs} started", flush=True)
        progress = tqdm(data_loader, desc=f"epoch {epoch_idx + 1}/{cfg.epochs}", leave=False)
        for a, p, n in progress:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            if cfg.loss_type == "triplet":
                emb_a = model.get_embedding(a)
                emb_p = model.get_embedding(p)
                if cfg.batch_hard_triplet and emb_a.size(0) > 1:
                    # Batch-hard negative mining from positives of other identities.
                    dist_mat = torch.cdist(emb_a, emb_p, p=2)
                    eye = torch.eye(dist_mat.size(0), device=dist_mat.device, dtype=torch.bool)
                    dist_mat = dist_mat.masked_fill(eye, float("inf"))
                    hard_idx = torch.argmin(dist_mat, dim=1)
                    emb_n = emb_p[hard_idx]
                else:
                    emb_n = model.get_embedding(n)
                loss = criterion(emb_a, emb_p, emb_n)
            else:
                emb_a = model.get_embedding(a)
                emb_p = model.get_embedding(p)
                emb_n = model.get_embedding(n)
                pos_labels = torch.ones(emb_a.size(0), device=device)
                neg_labels = torch.zeros(emb_a.size(0), device=device)
                loss_pos = criterion(emb_a, emb_p, pos_labels)
                loss_neg = criterion(emb_a, emb_n, neg_labels)
                loss = 0.5 * (loss_pos + loss_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            count += 1
            progress.set_postfix({"loss": f"{(running / max(1, count)):.4f}"})

        epoch_loss = running / max(1, count)
        train_losses.append(epoch_loss)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[train] epoch {epoch_idx + 1}/{cfg.epochs} done loss={epoch_loss:.6f} lr={current_lr:.7f}", flush=True)

        # Save per-epoch checkpoint to support restart after interruption.
        model_manager.save_checkpoint(
            model_path,
            optimizer=optimizer,
            epoch=epoch_idx + 1,
            loss=epoch_loss
        )

    print("[eval] selecting threshold on validation split...", flush=True)
    model_manager.save_checkpoint(model_path, optimizer=optimizer, epoch=cfg.epochs, loss=train_losses[-1] if train_losses else 0.0)

    model.eval()

    threshold, val_cls_metrics, val_auc = _find_best_threshold(model_manager, val_sketches, val_photos, preprocessor, device)

    print("[eval] scoring test split...", flush=True)
    test_y_true = []
    test_y_score = []
    n_test = len(test_sketches)
    for i in range(n_test):
        pos_dist = _distance(model_manager, test_sketches[i], test_photos[i], preprocessor, device)
        test_y_true.append(1)
        test_y_score.append(-pos_dist)

        j = random.randrange(n_test)
        while j == i:
            j = random.randrange(n_test)
        neg_dist = _distance(model_manager, test_sketches[i], test_photos[j], preprocessor, device)
        test_y_true.append(0)
        test_y_score.append(-neg_dist)

    y_true_np = np.array(test_y_true, dtype=np.int32)
    y_score_np = np.array(test_y_score, dtype=np.float32)
    y_pred_np = (y_score_np >= threshold).astype(np.int32)

    test_cls_metrics = _classification_metrics(y_true_np, y_pred_np)
    test_auc = _roc_auc_score(y_true_np, y_score_np)

    retrieval = _retrieval_metrics(
        model_manager,
        test_sketches,
        test_photos,
        preprocessor,
        device,
        top_k=cfg.top_k,
    )

    if cfg.reindex_after_train:
        print("[index] rebuilding embedding index from trained model...", flush=True)
        _reindex_with_trained_model(model_manager, preprocessor, db_path, index_path, metadata_path)

    elapsed = time.time() - start

    if cfg.gate_metric == "rank1_accuracy":
        gate_actual = retrieval["rank1_accuracy"]
    else:
        gate_actual = test_cls_metrics["accuracy"]

    passed_gate = bool(gate_actual >= cfg.target_accuracy)

    report = {
        "status": "passed" if passed_gate else "failed_target",
        "config": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "loss_type": cfg.loss_type,
            "target_accuracy": cfg.target_accuracy,
            "top_k": cfg.top_k,
            "seed": cfg.seed,
            "max_train_samples": cfg.max_train_samples,
            "max_eval_samples": cfg.max_eval_samples,
            "gate_metric": cfg.gate_metric,
            "resume_from_checkpoint": cfg.resume_from_checkpoint,
            "triplet_margin": cfg.triplet_margin,
            "use_augmentation": cfg.use_augmentation,
            "batch_hard_triplet": cfg.batch_hard_triplet,
        },
        "train": {
            "loss_curve": train_losses,
            "final_loss": train_losses[-1] if train_losses else None,
        },
        "validation": {
            **val_cls_metrics,
            "roc_auc": val_auc,
            "optimal_threshold": threshold,
        },
        "test": {
            **test_cls_metrics,
            "roc_auc": test_auc,
            **retrieval,
        },
        "gate": {
            "metric": cfg.gate_metric,
            "target": cfg.target_accuracy,
            "actual": gate_actual,
            "passed": passed_gate,
        },
        "artifacts": {
            "model_path": model_path,
            "report_path": report_path,
            "index_path": index_path,
            "metadata_path": metadata_path,
        },
        "timing": {
            "seconds": elapsed,
        },
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
