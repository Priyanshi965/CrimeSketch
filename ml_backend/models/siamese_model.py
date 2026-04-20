"""
Siamese Network Model for Sketch-to-Face Recognition

Uses InceptionResnetV1 pretrained on VGGFace2 (3.3M face images) as the backbone.
Face-discriminative features out of the box — significantly better than ImageNet
for cross-domain sketch-to-photo matching.

Falls back to ResNet50 + ImageNet normalization if facenet-pytorch is not installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("facenet-pytorch not found — falling back to ResNet50 backbone.")


class FeatureExtractor(nn.Module):
    """
    Face feature extractor.

    Primary: InceptionResnetV1(pretrained='vggface2') — 512-dim L2-normalized embeddings,
    pretrained on 3.3M face images for identity discrimination.

    Fallback: ResNet50 with ImageNet normalization applied internally.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim

        if FACENET_AVAILABLE:
            self.use_facenet = True
            if pretrained:
                try:
                    self.backbone = InceptionResnetV1(pretrained='vggface2')
                    print("Loaded InceptionResnetV1 pretrained on VGGFace2.")
                except Exception as e:
                    print(f"Warning: VGGFace2 download failed ({e}), using random init.")
                    self.backbone = InceptionResnetV1(pretrained=None)
            else:
                self.backbone = InceptionResnetV1(pretrained=None)

            # Small domain-adaptation head on top of 512-dim backbone output.
            # Bridges the sketch/photo domain gap without losing pretrained features.
            if embedding_dim == 512:
                self.adapter = nn.Sequential(
                    nn.Linear(512, 512, bias=False),
                    nn.BatchNorm1d(512),
                )
                # Initialize as near-identity so we don't destroy pretrained embeddings.
                nn.init.eye_(self.adapter[0].weight)
                nn.init.zeros_(self.adapter[1].weight)
                nn.init.ones_(self.adapter[1].weight)
            else:
                self.adapter = nn.Sequential(
                    nn.Linear(512, embedding_dim, bias=False),
                    nn.BatchNorm1d(embedding_dim),
                )

        else:
            self.use_facenet = False
            if pretrained:
                try:
                    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except Exception:
                    resnet50 = models.resnet50(weights=None)
            else:
                resnet50 = models.resnet50(weights=None)

            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])

            self.adapter = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )

            # ImageNet normalization constants — applied in forward for ResNet50.
            self.register_buffer(
                'imagenet_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'imagenet_std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) — grayscale repeated to 3 channels, normalized to [-1, 1]
        Returns:
            (batch, embedding_dim) — L2-normalized embeddings
        """
        if self.use_facenet:
            # InceptionResnetV1 expects [-1, 1] input, outputs L2-normalized 512-dim
            features = self.backbone(x)          # (batch, 512), already L2-normalized
            features = self.adapter(features)    # domain adaptation
        else:
            # ResNet50: input is [-1, 1], rescale to [0, 1] then apply ImageNet normalization
            x_01 = (x + 1.0) / 2.0
            x_norm = (x_01 - self.imagenet_mean) / self.imagenet_std
            features = self.backbone(x_norm)     # (batch, 2048, 1, 1)
            features = features.view(features.size(0), -1)
            features = self.adapter(features)    # (batch, embedding_dim)

        return F.normalize(features, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Siamese network for sketch-to-face matching."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.feature_extractor = FeatureExtractor(embedding_dim, pretrained)
        self.embedding_dim = embedding_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding1 = self.feature_extractor(x1)
        embedding2 = self.feature_extractor(x2)
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        return embedding1, embedding2, distance

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class TripletLoss(nn.Module):
    """
    Soft-margin triplet loss.
    Uses log(1 + exp(d_pos - d_neg)) instead of hard margin clamp —
    smoother gradients, better convergence on small datasets.
    """

    def __init__(self, margin: float = 0.3, soft: bool = True):
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        if self.soft:
            loss = torch.mean(F.softplus(pos_dist - neg_dist + self.margin))
        else:
            loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        loss = (label * torch.pow(distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return torch.mean(loss)


class ModelManager:
    """Manages model training, inference, and checkpoint saving/loading."""

    def __init__(self, model_type: str = 'siamese', embedding_dim: int = 512,
                 device: str = 'cpu', model_path: Optional[str] = None, pretrained: bool = True):
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.model_type = model_type

        if model_type == 'siamese':
            self.model = SiameseNetwork(embedding_dim=embedding_dim, pretrained=pretrained)
        elif model_type == 'feature_extractor':
            self.model = FeatureExtractor(embedding_dim=embedding_dim, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            self.load_checkpoint(model_path)

    def save_checkpoint(self, checkpoint_path: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'backbone': 'facenet_vggface2' if FACENET_AVAILABLE else 'resnet50',
            'epoch': epoch,
            'loss': loss,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
        except RuntimeError as e:
            print(f"Warning: checkpoint architecture mismatch, starting fresh. ({e})")
        return checkpoint

    def get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'siamese':
                embedding = self.model.get_embedding(image_tensor.to(self.device))
            else:
                embedding = self.model(image_tensor.to(self.device))
        return embedding.cpu().numpy()

    def get_embeddings_batch(self, image_tensors: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'siamese':
                embeddings = self.model.get_embedding(image_tensors.to(self.device))
            else:
                embeddings = self.model(image_tensors.to(self.device))
        return embeddings.cpu().numpy()
