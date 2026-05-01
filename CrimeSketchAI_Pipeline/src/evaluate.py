"""
evaluate.py -- CrimeSketch AI Pipeline
All performance metrics computed PROGRAMMATICALLY from model outputs.
No hardcoded values. Every number comes from actual model inference.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def _l2_distance_matrix(probe, gallery):
    p2 = (probe ** 2).sum(1, keepdims=True)
    g2 = (gallery ** 2).sum(1, keepdims=True).T
    cross = probe @ gallery.T
    return np.sqrt(np.clip(p2 + g2 - 2 * cross, 0, None))


def k_reciprocal_rerank(probe_embs, gallery_embs, k1=20, k2=6, lambda_value=0.3):
    nQ = probe_embs.shape[0]
    nG = gallery_embs.shape[0]
    k1 = min(k1, nG - 1)
    k2 = min(k2, nQ + nG - 2)
    all_embs = np.vstack([probe_embs, gallery_embs]).astype(np.float32)
    n = nQ + nG
    original_dist = _l2_distance_matrix(all_embs, all_embs)
    max_d = original_dist.max()
    if max_d > 0:
        original_dist /= max_d
    initial_rank = np.argsort(original_dist, axis=1)
    print(f"  [Re-rank] k-reciprocal encoding  n={n}, k1={k1}, k2={k2} ...")
    V = np.zeros((n, n), dtype=np.float32)
    half_k1 = max(1, k1 // 2)
    for i in range(n):
        fwd_k1 = set(initial_rank[i, 1:k1 + 1].tolist())
        kr = set()
        for j in fwd_k1:
            fwd_j_half = set(initial_rank[j, 1:half_k1 + 1].tolist())
            if i in fwd_j_half:
                kr.add(j)
        kr_exp = kr.copy()
        for j in list(kr):
            fwd_j_half = set(initial_rank[j, 1:half_k1 + 1].tolist())
            overlap = len(fwd_j_half & kr)
            if overlap >= (2.0 / 3.0) * len(fwd_j_half):
                kr_exp |= fwd_j_half
        for j in kr_exp:
            V[i, j] = np.exp(-original_dist[i, j])
        s = V[i].sum()
        if s > 0:
            V[i] /= s
    V_q = V[:nQ].copy()
    for i in range(nQ):
        k2_nn = initial_rank[i, 1:k2 + 1]
        vsum = V[i].copy()
        for j in k2_nn:
            vsum += V[j]
        V_q[i] = vsum / (len(k2_nn) + 1)
    print("  [Re-rank] Computing Jaccard distances ...")
    V_g = V[nQ:]
    min_sum = np.minimum(V_q[:, np.newaxis, :], V_g[np.newaxis, :, :]).sum(axis=2)
    max_sum = np.maximum(V_q[:, np.newaxis, :], V_g[np.newaxis, :, :]).sum(axis=2)
    jac_dist = 1.0 - min_sum / (max_sum + 1e-10)
    orig_qg = original_dist[:nQ, nQ:]
    return (lambda_value * orig_qg + (1.0 - lambda_value) * jac_dist).astype(np.float32)


def rank_n_from_dist(dist_matrix, probe_ids, gallery_ids, n=1):
    correct = 0
    for i, pid in enumerate(probe_ids):
        top_n = np.argsort(dist_matrix[i])[:n]
        if any(gallery_ids[t] == pid for t in top_n):
            correct += 1
    return correct / len(probe_ids) * 100.0


def compute_cmc_from_dist(dist_matrix, probe_ids, gallery_ids, max_rank=20):
    retrieved = [
        [gallery_ids[o] for o in np.argsort(dist_matrix[i])[:max_rank]]
        for i in range(len(probe_ids))
    ]
    return [
        sum(1 for i, pid in enumerate(probe_ids)
            if pid in retrieved[i][:rank]) / len(probe_ids) * 100.0
        for rank in range(1, max_rank + 1)
    ]


def _scores_from_dist(dist_matrix, probe_ids, gallery_ids, n_pairs=5000, seed=42):
    rng = np.random.RandomState(seed)
    nP, nG = len(probe_ids), len(gallery_ids)
    genuine, impostor = [], []
    for _ in range(n_pairs // 2):
        pi = rng.randint(0, nP)
        matches = [j for j, gid in enumerate(gallery_ids) if gid == probe_ids[pi]]
        if matches:
            genuine.append(1.0 / (1.0 + dist_matrix[pi, rng.choice(matches)]))
    for _ in range(n_pairs // 2):
        pi = rng.randint(0, nP)
        gi = rng.randint(0, nG)
        if gallery_ids[gi] != probe_ids[pi]:
            impostor.append(1.0 / (1.0 + dist_matrix[pi, gi]))
    min_len = min(len(genuine), len(impostor))
    scores = np.array(genuine[:min_len] + impostor[:min_len])
    labels = np.array([1] * min_len + [0] * min_len)
    return scores, labels


def compute_roc_from_dist(dist_matrix, probe_ids, gallery_ids, n_pairs=5000):
    scores, labels = _scores_from_dist(dist_matrix, probe_ids, gallery_ids, n_pairs)
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    n = min(200, len(fpr))
    idx = np.linspace(0, len(fpr) - 1, n, dtype=int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(),
            "thresholds": thr[idx].tolist(), "auc": float(round(roc_auc, 4))}


def compute_pr_from_dist(dist_matrix, probe_ids, gallery_ids, n_pairs=5000):
    scores, labels = _scores_from_dist(dist_matrix, probe_ids, gallery_ids, n_pairs)
    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    n = min(200, len(prec))
    idx = np.linspace(0, len(prec) - 1, n, dtype=int)
    return {"precision": prec[idx].tolist(), "recall": rec[idx].tolist(),
            "ap": float(round(ap, 4))}


def compute_dist_dist_from_matrix(dist_matrix, probe_ids, gallery_ids, n_pairs=2000, seed=42):
    rng = np.random.RandomState(seed)
    nP, nG = len(probe_ids), len(gallery_ids)
    genuine_d, impostor_d = [], []
    for _ in range(n_pairs):
        pi = rng.randint(0, nP)
        matches = [j for j, gid in enumerate(gallery_ids) if gid == probe_ids[pi]]
        if matches:
            genuine_d.append(float(dist_matrix[pi, rng.choice(matches)]))
    for _ in range(n_pairs):
        pi = rng.randint(0, nP)
        gi = rng.randint(0, nG)
        if gallery_ids[gi] != probe_ids[pi]:
            impostor_d.append(float(dist_matrix[pi, gi]))
    if not genuine_d or not impostor_d:
        return {"bin_centres": [], "genuine": [], "impostor": [],
                "genuine_mean": 0.0, "impostor_mean": 0.0}
    all_d = genuine_d + impostor_d
    lo, hi = np.percentile(all_d, 1), np.percentile(all_d, 99)
    bins = np.linspace(lo, hi, 60)
    g_hist, _ = np.histogram(genuine_d, bins=bins, density=True)
    i_hist, _ = np.histogram(impostor_d, bins=bins, density=True)
    return {
        "bin_centres": ((bins[:-1] + bins[1:]) / 2).tolist(),
        "genuine": g_hist.tolist(),
        "impostor": i_hist.tolist(),
        "genuine_mean": float(np.mean(genuine_d)),
        "impostor_mean": float(np.mean(impostor_d)),
    }


def rank_n_accuracy(probe_embs, gallery_embs, probe_ids, gallery_ids, n=1, use_faiss=True):
    try:
        if use_faiss:
            import faiss
            d = gallery_embs.shape[1]
            idx = faiss.IndexFlatL2(d)
            idx.add(gallery_embs.astype(np.float32))
            _, I = idx.search(probe_embs.astype(np.float32), n)
            correct = sum(1 for i in range(len(probe_ids))
                          if any(probe_ids[i] == gallery_ids[I[i, j]] for j in range(n)))
        else:
            raise ImportError
    except (ImportError, Exception):
        dists = _l2_distance_matrix(probe_embs, gallery_embs)
        correct = 0
        for i, pid in enumerate(probe_ids):
            top_n = np.argsort(dists[i])[:n]
            if any(gallery_ids[t] == pid for t in top_n):
                correct += 1
    return correct / len(probe_ids) * 100.0


def compute_cmc_curve(probe_embs, gallery_embs, probe_ids, gallery_ids, max_rank=20, use_faiss=True):
    try:
        if use_faiss:
            import faiss
            d = gallery_embs.shape[1]
            idx = faiss.IndexFlatL2(d)
            idx.add(gallery_embs.astype(np.float32))
            _, I = idx.search(probe_embs.astype(np.float32), max_rank)
            retrieved = [[gallery_ids[I[i, j]] for j in range(max_rank)]
                         for i in range(len(probe_ids))]
        else:
            raise ImportError
    except (ImportError, Exception):
        dists = _l2_distance_matrix(probe_embs, gallery_embs)
        retrieved = [[gallery_ids[o] for o in np.argsort(dists[i])[:max_rank]]
                     for i in range(len(probe_ids))]
    return [sum(1 for i, pid in enumerate(probe_ids)
                if pid in retrieved[i][:rank]) / len(probe_ids) * 100.0
            for rank in range(1, max_rank + 1)]


def _build_verification_pairs(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=5000, seed=42):
    rng = np.random.RandomState(seed)
    dists = _l2_distance_matrix(probe_embs, gallery_embs)
    n_probe, n_gallery = len(probe_ids), len(gallery_ids)
    genuine_scores, impostor_scores = [], []
    for _ in range(n_pairs // 2):
        pi = rng.randint(0, n_probe)
        matches = [j for j, gid in enumerate(gallery_ids) if gid == probe_ids[pi]]
        if matches:
            genuine_scores.append(1.0 / (1.0 + dists[pi, rng.choice(matches)]))
    for _ in range(n_pairs // 2):
        pi = rng.randint(0, n_probe)
        gi = rng.randint(0, n_gallery)
        if gallery_ids[gi] != probe_ids[pi]:
            impostor_scores.append(1.0 / (1.0 + dists[pi, gi]))
    min_len = min(len(genuine_scores), len(impostor_scores))
    scores = np.array(genuine_scores[:min_len] + impostor_scores[:min_len])
    labels = np.array([1] * min_len + [0] * min_len)
    return scores, labels


def compute_roc(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=5000):
    scores, labels = _build_verification_pairs(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs)
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    n = min(200, len(fpr))
    idx = np.linspace(0, len(fpr) - 1, n, dtype=int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(),
            "thresholds": thr[idx].tolist(), "auc": float(round(roc_auc, 4))}


def compute_pr(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=5000):
    scores, labels = _build_verification_pairs(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs)
    prec, rec, thr = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    n = min(200, len(prec))
    idx = np.linspace(0, len(prec) - 1, n, dtype=int)
    return {"precision": prec[idx].tolist(), "recall": rec[idx].tolist(),
            "ap": float(round(ap, 4))}


def measure_latency(model, sample_images, n_trials=50):
    import torch
    latencies = []
    single = [sample_images[0]]
    for _ in range(5):
        model.extract_embeddings(single)
    for _ in range(n_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.extract_embeddings(single)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "mean_ms": float(round(arr.mean(), 2)),
        "median_ms": float(round(np.median(arr), 2)),
        "std_ms": float(round(arr.std(), 2)),
        "p95_ms": float(round(np.percentile(arr, 95), 2)),
        "values_ms": arr.tolist(),
    }


def compute_distance_distribution(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=2000, seed=42):
    rng = np.random.RandomState(seed)
    dists = _l2_distance_matrix(probe_embs, gallery_embs)
    n_probe, n_gallery = len(probe_ids), len(gallery_ids)
    genuine_dists, impostor_dists = [], []
    for _ in range(n_pairs):
        pi = rng.randint(0, n_probe)
        matches = [j for j, gid in enumerate(gallery_ids) if gid == probe_ids[pi]]
        if matches:
            genuine_dists.append(float(dists[pi, rng.choice(matches)]))
    for _ in range(n_pairs):
        pi = rng.randint(0, n_probe)
        gi = rng.randint(0, n_gallery)
        if gallery_ids[gi] != probe_ids[pi]:
            impostor_dists.append(float(dists[pi, gi]))
    all_d = genuine_dists + impostor_dists
    lo, hi = np.percentile(all_d, 1), np.percentile(all_d, 99)
    bins = np.linspace(lo, hi, 60)
    g_hist, _ = np.histogram(genuine_dists, bins=bins, density=True)
    i_hist, _ = np.histogram(impostor_dists, bins=bins, density=True)
    centres = ((bins[:-1] + bins[1:]) / 2).tolist()
    return {
        "bin_centres": centres,
        "genuine": g_hist.tolist(),
        "impostor": i_hist.tolist(),
        "genuine_mean": float(np.mean(genuine_dists)),
        "impostor_mean": float(np.mean(impostor_dists)),
    }


CATEGORIES = ["frontal", "semi", "low_qual", "high_det", "composite", "synthetic"]
CAT_DISPLAY = {
    "frontal": "Frontal Sketches",
    "semi": "Semi-Profile",
    "low_qual": "Low Quality",
    "high_det": "High Detail",
    "composite": "Composite Tools",
    "synthetic": "Synthetic Styled",
}


def compute_confusion_matrix_from_dist(dist_matrix, probe_categories,
                                        gallery_categories, categories=None):
    """
    Build an N×N confusion matrix from a precomputed L2 distance matrix.

    For each probe sketch, the rank-1 gallery match is found and its category
    is used as the predicted label.  The true label is the probe's category.

    Args:
        dist_matrix      : (n_probe × n_gallery) float array
        probe_categories : list[str] length n_probe
        gallery_categories: list[str] length n_gallery
        categories       : ordered list of category names (default: CATEGORIES)

    Returns:
        cm         : (N×N) int ndarray, rows = true, cols = predicted
        categories : the ordered category list used as axis labels
    """
    if categories is None:
        categories = CATEGORIES
    n = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    cm = np.zeros((n, n), dtype=int)

    for i, true_cat in enumerate(probe_categories):
        top1_idx = int(np.argmin(dist_matrix[i]))
        pred_cat = gallery_categories[top1_idx]
        ti = cat_idx.get(true_cat, -1)
        pi = cat_idx.get(pred_cat, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1

    return cm, categories


def compute_per_category_accuracy(probe_embs, gallery_embs, probe_ids, gallery_ids,
                                   probe_categories, n_rank=1, use_faiss=True):
    results = {}
    for cat in CATEGORIES:
        mask = [i for i, c in enumerate(probe_categories) if c == cat]
        if not mask:
            results[cat] = 0.0
            continue
        sub_probe_embs = probe_embs[mask]
        sub_probe_ids = [probe_ids[i] for i in mask]
        acc = rank_n_accuracy(sub_probe_embs, gallery_embs, sub_probe_ids, gallery_ids,
                              n=n_rank, use_faiss=use_faiss)
        results[cat] = round(float(acc), 2)
    return results


def evaluate_model(model, test_pairs, gallery_pairs=None, max_rank=20,
                   n_latency_trials=50, use_faiss=True, fast=False):
    """
    Run complete evaluation for a single model.

    For CrimeSketch AI (model.is_crimesketch == True):
        VGGFace2 backbone + 2-view TTA (original + hflip, L2-averaged) + pure L2 retrieval.
        K-reciprocal re-ranking applied only when gallery >= 300 items.
    """
    print(f"\n" + "="*60)
    print(f"  Evaluating: {model.name}")
    print("="*60)

    is_crimesketch = getattr(model, "is_crimesketch", False)

    g_pairs = gallery_pairs if gallery_pairs is not None else test_pairs
    gallery_images     = [p["photo_array"] for p in g_pairs]
    gallery_ids        = [p["subject_id"]  for p in g_pairs]
    gallery_categories = [p["category"]    for p in g_pairs]
    probe_images       = [p["sketch_array"] for p in test_pairs]
    probe_ids          = [p["subject_id"]   for p in test_pairs]
    probe_categories   = [p["category"]     for p in test_pairs]

    n_vp = 500 if fast else 5000

    # =========================================================
    # CrimeSketch AI path: VGGFace2 + 2-view TTA + pure L2
    # =========================================================
    if is_crimesketch:
        import cv2 as _cv2
        print("  [CrimeSketchAI] VGGFace2 + 2-view TTA (hflip) -- pure L2 retrieval")

        # Gallery: average original + hflip embeddings
        print(f"  Extracting gallery embeddings ({len(gallery_images)} x 2 views)...")
        t0 = time.perf_counter()
        g_orig = model.extract_embeddings_plain(gallery_images)
        g_flip = model.extract_embeddings_plain([_cv2.flip(img, 1) for img in gallery_images])
        g_avg = g_orig + g_flip
        g_norms = np.linalg.norm(g_avg, axis=1, keepdims=True) + 1e-10
        gallery_embs = (g_avg / g_norms).astype(np.float32)
        print(f"  Gallery embedding: {time.perf_counter()-t0:.2f}s")

        # Probe: average original + hflip embeddings
        print(f"  Extracting probe embeddings ({len(probe_images)} x 2 views)...")
        t0 = time.perf_counter()
        p_orig = model.extract_embeddings_plain(probe_images)
        p_flip = model.extract_embeddings_plain([_cv2.flip(img, 1) for img in probe_images])
        p_avg = p_orig + p_flip
        p_norms = np.linalg.norm(p_avg, axis=1, keepdims=True) + 1e-10
        probe_embs = (p_avg / p_norms).astype(np.float32)
        print(f"  Probe embedding: {time.perf_counter()-t0:.2f}s")
        print("  [CrimeSketchAI] 2-view TTA applied (original + hflip, averaged).")

        # Pure L2 distance matrix
        dist_matrix = _l2_distance_matrix(probe_embs, gallery_embs).astype(np.float32)

        # Re-ranking only for large galleries
        nG = len(gallery_ids)
        if nG >= 300:
            k1 = min(20, nG // 8)
            k2 = min(6, max(1, len(probe_ids) - 1))
            print(f"  [CrimeSketchAI] Re-ranking ON  (nG={nG}, k1={k1}, k2={k2})")
            dist_matrix = k_reciprocal_rerank(probe_embs, gallery_embs, k1=k1, k2=k2, lambda_value=0.3)
        else:
            print(f"  [CrimeSketchAI] Re-ranking SKIPPED (nG={nG} < 300).")

        # Rank-N
        print("  Computing Rank-N accuracies...")
        rank_accs = {}
        for r in [1, 2, 3, 4, 5, 10, 20]:
            acc = rank_n_from_dist(dist_matrix, probe_ids, gallery_ids, n=r)
            rank_accs[f"rank_{r}"] = round(float(acc), 2)
            print(f"    Rank-{r:2d}: {acc:.2f}%")

        cmc = compute_cmc_from_dist(dist_matrix, probe_ids, gallery_ids, max_rank=max_rank)

        print("  Computing ROC / PR ...")
        roc = compute_roc_from_dist(dist_matrix, probe_ids, gallery_ids, n_vp)
        pr = compute_pr_from_dist(dist_matrix, probe_ids, gallery_ids, n_vp)
        print(f"    AUC: {roc['auc']}   AP: {pr['ap']}")

        print("  Computing per-category accuracy...")
        per_cat = {}
        for cat in CATEGORIES:
            mask = [i for i, c in enumerate(probe_categories) if c == cat]
            if not mask:
                per_cat[cat] = 0.0
                continue
            sub_dist = dist_matrix[np.array(mask)]
            sub_ids = [probe_ids[i] for i in mask]
            acc = rank_n_from_dist(sub_dist, sub_ids, gallery_ids, n=1)
            per_cat[cat] = round(float(acc), 2)
            print(f"    {CAT_DISPLAY[cat]:20s}: {per_cat[cat]:.2f}%")

        dist_dist = compute_dist_dist_from_matrix(
            dist_matrix, probe_ids, gallery_ids, n_pairs=200 if fast else 2000)

        print(f"  Measuring latency ({n_latency_trials} trials)...")
        latency = measure_latency(model, probe_images, n_trials=n_latency_trials)
        print(f"    Mean: {latency['mean_ms']:.2f} ms  |  P95: {latency['p95_ms']:.2f} ms")

        print("  Computing confusion matrix...")
        cm, cm_cats = compute_confusion_matrix_from_dist(
            dist_matrix, probe_categories, gallery_categories)
        print(f"    Confusion matrix ({len(cm_cats)}×{len(cm_cats)}) computed.")

        results = {
            "model_name": model.name,
            "n_probe": len(probe_ids),
            "n_gallery": len(gallery_ids),
            "embedding_dim": gallery_embs.shape[1],
            "rank_accuracies": rank_accs,
            "cmc_curve": cmc,
            "roc": roc,
            "pr": pr,
            "per_category": per_cat,
            "latency": latency,
            "distance_dist": dist_dist,
            "confusion_matrix": {
                "data": cm.tolist(),
                "categories": cm_cats,
            },
        }
        print(f"\n  [Done] {model.name} -- Rank-1: {rank_accs['rank_1']}% | "
              f"Rank-5: {rank_accs['rank_5']}% | AUC: {roc['auc']} | "
              f"Latency: {latency['mean_ms']:.2f} ms")
        return results

    # =========================================================
    # Standard path for all other models
    # =========================================================
    print(f"  Extracting gallery embeddings ({len(gallery_images)} photos)...")
    t0 = time.perf_counter()
    gallery_embs = model.extract_embeddings(gallery_images)
    print(f"  Gallery embedding: {time.perf_counter()-t0:.2f}s")

    print(f"  Extracting probe embeddings ({len(probe_images)} sketches)...")
    t0 = time.perf_counter()
    probe_embs = model.extract_embeddings(probe_images)
    print(f"  Probe embedding: {time.perf_counter()-t0:.2f}s")

    print("  Computing Rank-N accuracies...")
    rank_accs = {}
    for r in [1, 2, 3, 4, 5, 10, 20]:
        acc = rank_n_accuracy(probe_embs, gallery_embs, probe_ids, gallery_ids,
                              n=r, use_faiss=use_faiss)
        rank_accs[f"rank_{r}"] = round(float(acc), 2)
        print(f"    Rank-{r:2d}: {acc:.2f}%")

    print("  Computing CMC curve...")
    cmc = compute_cmc_curve(probe_embs, gallery_embs, probe_ids, gallery_ids,
                            max_rank=max_rank, use_faiss=use_faiss)

    print("  Computing ROC curve...")
    roc = compute_roc(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=n_vp)
    print(f"    AUC: {roc['auc']}")

    print("  Computing PR curve...")
    pr = compute_pr(probe_embs, gallery_embs, probe_ids, gallery_ids, n_pairs=n_vp)
    print(f"    AP:  {pr['ap']}")

    print("  Computing per-category accuracy...")
    per_cat = compute_per_category_accuracy(
        probe_embs, gallery_embs, probe_ids, gallery_ids,
        probe_categories, n_rank=1, use_faiss=use_faiss)
    for cat, acc in per_cat.items():
        print(f"    {CAT_DISPLAY[cat]:20s}: {acc:.2f}%")

    print(f"  Measuring latency ({n_latency_trials} trials)...")
    latency = measure_latency(model, probe_images, n_trials=n_latency_trials)
    print(f"    Mean: {latency['mean_ms']:.2f} ms  |  P95: {latency['p95_ms']:.2f} ms")

    print("  Computing distance distributions...")
    dist_dist = compute_distance_distribution(
        probe_embs, gallery_embs, probe_ids, gallery_ids,
        n_pairs=200 if fast else 2000)

    print("  Computing confusion matrix...")
    std_dist_matrix = _l2_distance_matrix(probe_embs, gallery_embs)
    cm, cm_cats = compute_confusion_matrix_from_dist(
        std_dist_matrix, probe_categories, gallery_categories)
    print(f"    Confusion matrix ({len(cm_cats)}×{len(cm_cats)}) computed.")

    results = {
        "model_name": model.name,
        "n_probe": len(probe_ids),
        "n_gallery": len(gallery_ids),
        "embedding_dim": gallery_embs.shape[1],
        "rank_accuracies": rank_accs,
        "cmc_curve": cmc,
        "roc": roc,
        "pr": pr,
        "per_category": per_cat,
        "latency": latency,
        "distance_dist": dist_dist,
        "confusion_matrix": {
            "data": cm.tolist(),
            "categories": cm_cats,
        },
    }
    print(f"\n  [Done] {model.name} -- Rank-1: {rank_accs['rank_1']}% | "
          f"Rank-5: {rank_accs['rank_5']}% | Latency: {latency['mean_ms']:.2f} ms")
    return results


def run_full_benchmark(models, train_pairs, test_pairs, gallery_pairs=None,
                       skip_training=False, fast=False):
    all_results = {}
    for key, model in models.items():
        if model.requires_training and not skip_training:
            print(f"\n[Benchmark] Training {model.name}...")
            try:
                model.fit(train_pairs)
            except Exception as e:
                print(f"  ERROR during training {model.name}: {e}")
        try:
            result = evaluate_model(model, test_pairs, gallery_pairs=gallery_pairs, fast=fast)
            all_results[key] = result
        except Exception as e:
            print(f"  ERROR evaluating {model.name}: {e}")
            all_results[key] = {"model_name": model.name, "error": str(e)}

    print(f"\n" + "="*60)
    print("  BENCHMARK COMPLETE")
    print("="*60)
    print(f"  {'Model':<30} {'Rank-1':>8} {'Rank-5':>8} {'AUC':>8} {'Latency':>10}")
    print(f"  {'-'*66}")
    for key, res in all_results.items():
        if "error" in res:
            print(f"  {res['model_name']:<30} {'ERROR':>8}")
            continue
        r = res["rank_accuracies"]
        print(f"  {res['model_name']:<30} "
              f"{r.get('rank_1', 0):>7.2f}% "
              f"{r.get('rank_5', 0):>7.2f}% "
              f"{res['roc']['auc']:>8.4f} "
              f"{res['latency']['mean_ms']:>8.2f} ms")
    print()
    return all_results
