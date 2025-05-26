import torch
import numpy as np
from typing import List, Tuple, Union
from core.third_party.torchSIFT.src.torchsift.ransac.ransac import ransac
import math

def is_cuda_oom(e: RuntimeError) -> bool:
    """Check if the RuntimeError was caused by CUDA out-of-memory."""
    return isinstance(e, RuntimeError) and 'CUDA out of memory' in str(e)


def safe_empty_cache():
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"[⚠️] Skipped empty_cache() due to illegal access: {e}")

def filter_keypoints(feat: dict, score_thresh: float) -> Tuple[dict, np.ndarray]:
    """
    Filters keypoints based on score threshold.

    Args:
        feat: Dict with keys ['descriptors', 'keypoints', 'scores', 'image_size']
        score_thresh: Score filtering threshold (e.g., 0.2)

    Returns:
        filtered_feat: Same keys, with filtered tensors
        keep_inds: Original indices of retained keypoints (numpy array)
    """
    mask = feat['scores'] >= score_thresh
    keep_inds = torch.where(mask)[0]

    filtered = {
        'descriptors': feat['descriptors'][:, mask],
        'keypoints': feat['keypoints'][mask],
        'scores': feat['scores'][mask],
        'image_size': feat['image_size'],
    }
    return filtered, keep_inds.cpu().numpy()

def dynamic_score_threshold(n_kpts: int, base_thresh: float = 0.2, gamma: float = 0.005) -> float:
    """
    Exponential growth of threshold from 0 at 50 keypoints
    to base_thresh around 500 keypoints.
    """
    if n_kpts <= 50:
        return 0.0
    elif n_kpts >= 500:
        return base_thresh
    else:
        x = n_kpts - 50
        growth = 1.0 - math.exp(-gamma * x)
        return base_thresh * growth

def match_query_to_database(
    q_feat: dict,
    db_feats_list: List[dict],
    db_names_list: List[str],
    local_feature_matcher,
    device,
    feature_score_threshold: float = 0.2,
    threshold: int = 20,
    max_safe_product: int = 10_000_000,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Match a query image feature to a batch of database features using LightGlue.
    Supports dynamic score filtering, batch splitting, and CPU fallback.
    """
    
    def safe_match(
        q_feat: dict,
        db_feats: List[dict],
        db_names: List[str],
        device
    ) -> Union[None, Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        # Prepare output containers
        pts0_idx_list, pts1_idx_list, scores_list, valid_db_names = [], [], [], []

        # 1. Filter query features dynamically
        n_q = q_feat['scores'].shape[0]
        q_thresh = dynamic_score_threshold(n_q, base_thresh=feature_score_threshold)
        q_filtered, query_keep = filter_keypoints(q_feat, q_thresh)
        if q_filtered['keypoints'].shape[0] == 0:
            return [], [], [], []

        # 2. Collect valid DB features (filtered and non-empty)
        filtered_feats = []
        filtered_names = []
        keep_inds_list = []
        for feat, name in zip(db_feats, db_names):
            n_db = feat['scores'].shape[0]
            db_thresh = dynamic_score_threshold(n_db, base_thresh=feature_score_threshold)
            db_filtered, db_keep = filter_keypoints(feat, db_thresh)
            if db_filtered['keypoints'].shape[0] == 0:
                continue
            filtered_feats.append(db_filtered)
            filtered_names.append(name)
            keep_inds_list.append(db_keep)

        B = len(filtered_feats)
        if B == 0:
            return [], [], [], []

        desc_dim = q_filtered['descriptors'].shape[0]
        max_k = max(f['descriptors'].shape[1] for f in filtered_feats)

        # 3. Build LightGlue tensors
        desc0 = q_filtered['descriptors'].unsqueeze(0).expand(B, -1, -1).to(device)
        kpts0 = q_filtered['keypoints'].unsqueeze(0).expand(B, -1, -1).to(device)
        scores0 = q_filtered['scores'].unsqueeze(0).expand(B, -1).to(device)
        img0 = q_filtered['image_size'].unsqueeze(0).expand(B, -1).to(device)

        desc1 = torch.zeros((B, desc_dim, max_k), device=device)
        kpts1 = torch.zeros((B, max_k, 2), device=device)
        scores1 = torch.zeros((B, max_k), device=device)
        img1 = torch.zeros((B, 2), device=device)

        for i, feat in enumerate(filtered_feats):
            n = feat['keypoints'].shape[0]
            desc1[i, :, :n] = feat['descriptors']
            kpts1[i, :n] = feat['keypoints']
            scores1[i, :n] = feat['scores']
            img1[i] = feat['image_size']

        pred = {
            'descriptors0': desc0,
            'keypoints0': kpts0,
            'keypoint_scores0': scores0,
            'image_size0': img0,
            'descriptors1': desc1,
            'keypoints1': kpts1,
            'keypoint_scores1': scores1,
            'image_size1': img1,
        }

        # 4. Run matching
        with torch.inference_mode():
            out = local_feature_matcher(pred)

        matches = out['matches0'].cpu().numpy()
        m_scores = out['matching_scores0'].cpu().numpy()

        # 5. Collect verified matches
        for i, match in enumerate(matches):
            max_valid_idx = keep_inds_list[i].shape[0]

            ui, vi, si = [], [], []
            for u, v in enumerate(match):
                if v != -1:
                    ui.append(u)
                    vi.append(v)
                    si.append(m_scores[i, u])

            # Filter out any v that exceeds allowed index
            filtered = [(u, v, s) for u, v, s in zip(ui, vi, si) if v < max_valid_idx]

            if len(filtered) >= threshold:
                u_valid, v_valid, s_valid = zip(*filtered)
                pts0_idx_list.append(query_keep[list(u_valid)])
                pts1_idx_list.append(keep_inds_list[i][list(v_valid)])
                scores_list.append(np.array(s_valid, dtype=np.float32))
                valid_db_names.append(filtered_names[i])

        return valid_db_names, pts0_idx_list, pts1_idx_list, scores_list

    # Entry point: check empty input
    if not db_feats_list:
        return [], [], [], []

    # Split large batches
    B0 = len(db_feats_list)
    max_k0 = max(f['descriptors'].shape[1] for f in db_feats_list)
    if B0 * max_k0 > max_safe_product:
        mid = B0 // 2
        out1 = match_query_to_database(q_feat, db_feats_list[:mid], db_names_list[:mid],
                                       local_feature_matcher, device,
                                       feature_score_threshold, threshold, max_safe_product)
        out2 = match_query_to_database(q_feat, db_feats_list[mid:], db_names_list[mid:],
                                       local_feature_matcher, device,
                                       feature_score_threshold, threshold, max_safe_product)
        return ([*out1[0], *out2[0]], [*out1[1], *out2[1]], [*out1[2], *out2[2]], [*out1[3], *out2[3]])

    # Try GPU match, fallback to CPU if OOM
    try:
        return safe_match(q_feat, db_feats_list, db_names_list, device)
    except RuntimeError as e:
        if is_cuda_oom(e) and B0 == 1:
            safe_empty_cache()
            return safe_match(q_feat, db_feats_list, db_names_list, 'cpu')
        raise


def ransac_filter(
    pts0_idx_list: List[np.ndarray],
    pts1_idx_list: List[np.ndarray],
    kpts0_list: List[np.ndarray],
    kpts1_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    device,
    threshold: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    MAX_SAFE_TOTAL = 12_000_000  # batch * max_len^2 upper limit (tuned for 24GB GPU)

    batch_size = len(pts0_idx_list)
    valid_flags = np.zeros(batch_size, dtype=bool)
    inliers0 = [None] * batch_size
    inliers1 = [None] * batch_size
    inlier_scores = [None] * batch_size

    def run_single_ransac_on_cpu(idx: int):
        pts0_np = kpts0_list[idx][pts0_idx_list[idx]]
        pts1_np = kpts1_list[idx][pts1_idx_list[idx]]
        if pts0_np.shape[0] < 4:
            return
        pts0 = torch.tensor(pts0_np, dtype=torch.float32, device='cpu').unsqueeze(0)
        pts1 = torch.tensor(pts1_np, dtype=torch.float32, device='cpu').unsqueeze(0)
        mask = torch.ones((1, pts0_np.shape[0], pts0_np.shape[0]), dtype=torch.float32)
        with torch.no_grad():
            _, inl_mask, _ = ransac(pts0, pts1, mask)
        diag_mask = inl_mask.squeeze().diag().bool().numpy()
        num_inliers = diag_mask.sum()
        if num_inliers >= threshold:
            valid_flags[idx] = True
            inliers0[idx] = pts0_idx_list[idx][diag_mask]
            inliers1[idx] = pts1_idx_list[idx][diag_mask]
            inlier_scores[idx] = scores_list[idx][diag_mask]

    def recursive_ransac(batch_indices: List[int]):
        if not batch_indices:
            return

        n = len(batch_indices)
        max_len = max(len(pts0_idx_list[i]) for i in batch_indices)
        if max_len < 4:
            return

        if n * (max_len ** 2) > MAX_SAFE_TOTAL:
            if n == 1:
                run_single_ransac_on_cpu(batch_indices[0])
                return
            mid = n // 2
            recursive_ransac(batch_indices[:mid])
            safe_empty_cache()
            recursive_ransac(batch_indices[mid:])
            safe_empty_cache()
            return

        try:
            pts0 = torch.zeros((n, max_len, 2), dtype=torch.float32, device=device)
            pts1 = torch.zeros((n, max_len, 2), dtype=torch.float32, device=device)
            mask = torch.zeros((n, max_len, max_len), dtype=torch.float32, device=device)

            for i, idx in enumerate(batch_indices):
                n_i = len(pts0_idx_list[idx])
                if n_i == 0:
                    continue
                pts0[i, :n_i] = torch.tensor(kpts0_list[idx][pts0_idx_list[idx]], dtype=torch.float32)
                pts1[i, :n_i] = torch.tensor(kpts1_list[idx][pts1_idx_list[idx]], dtype=torch.float32)
                mask[i, :n_i, :n_i] = 1.0

            _, inl_mask, _ = ransac(pts0, pts1, mask)
            diag_masks = torch.diagonal(inl_mask, dim1=-2, dim2=-1)
            sizes = diag_masks.sum(-1)

            for i, idx in enumerate(batch_indices):
                n_i = len(pts0_idx_list[idx])
                if n_i == 0:
                    continue
                if sizes[i] >= threshold:
                    diag_mask_i = diag_masks[i, :n_i].bool().cpu().numpy()
                    valid_flags[idx] = True
                    inliers0[idx] = pts0_idx_list[idx][diag_mask_i]
                    inliers1[idx] = pts1_idx_list[idx][diag_mask_i]
                    inlier_scores[idx] = scores_list[idx][diag_mask_i]

            del pts0, pts1, mask, inl_mask, diag_masks, sizes
            safe_empty_cache()

        except RuntimeError as e:
            if is_cuda_oom(e) or "out of memory" in str(e):
                safe_empty_cache()

            if n == 1:
                run_single_ransac_on_cpu(batch_indices[0])
                return

            mid = n // 2
            recursive_ransac(batch_indices[:mid])
            safe_empty_cache()
            recursive_ransac(batch_indices[mid:])
            safe_empty_cache()

    recursive_ransac(list(range(batch_size)))
    return valid_flags, inliers0, inliers1, inlier_scores


