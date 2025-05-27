import numpy as np
import torch
from typing import Dict, Tuple, List, Any, Callable

def search_vpr_topk_candidates(
    query_feature: np.ndarray,
    all_map_features: Dict[str, Tuple[np.ndarray, List[str]]],
    top_k: int = 5,
    device: str = "cuda"
) -> List[Tuple[str, str, float]]:
    """
    Search the Top-K most similar reference images for a query image over all map partitions.

    Args:
        query_feature (np.ndarray): Query global descriptor, shape (D,).
        all_map_features (Dict[str, Tuple[np.ndarray, List[str]]]): 
            key = "place__building__floor",
            value = (features array [N, D], names list of length N).
        top_k (int): Number of overall top-K matches to return.
        device (str): Torch device ('cuda' or 'cpu').

    Returns:
        List[Tuple[str, str, float]]: 
            Each tuple is (map_key, ref_image_name, similarity_score), sorted descending by score.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    qfeat = torch.tensor(query_feature, dtype=torch.float32, device=device)
    qfeat = torch.nn.functional.normalize(qfeat, dim=0)

    candidates = []
    for map_key, (features, names) in all_map_features.items():
        feats = torch.tensor(features, dtype=torch.float32, device=device)
        feats = torch.nn.functional.normalize(feats, dim=1)
        sims = feats @ qfeat  # [N]
        top_scores, top_indices = torch.topk(sims, k=min(top_k, feats.shape[0]))
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            candidates.append((map_key, names[idx], score))
    # Global sort
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]

def fetch_candidates_data(
    all_colmap_models: Dict[str, Dict[str, Any]],
    local_feat_paths: Dict[str, str],
    top_candidates: List[tuple],
    load_local_features_fn: Callable[[str, List[str]], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Batch-fetches map data (COLMAP frames and local features) for the given list of VPR candidates.

    Args:
        all_colmap_models (Dict[str, Dict[str, Any]]): Mapping from map key ("place__building__floor")
            to a dictionary of frames indexed by image name.
        local_feat_paths (Dict[str, str]): Mapping from map key to the local feature HDF5 file path.
        top_candidates (List[tuple]): List of (map_key, ref_img_name, score) VPR candidates.
        load_local_features_fn (Callable): Function to load local features given an HDF5 file path and image name list.

    Returns:
        Dict[str, Dict[str, Any]]: 
            Dictionary mapping each reference image name to a dictionary containing:
                - "frame" (dict): Frame information from COLMAP.
                - "local_feat" (dict): Local features for the reference image.
                - "map_key" (str): Map key/domain this image belongs to.
                - "score" (float): VPR similarity score for this candidate.
    """
    # Group candidates by map key for efficient batch loading.
    groups = {}
    for map_key, ref_name, score in top_candidates:
        groups.setdefault(map_key, []).append((ref_name, score))

    candidates_data = {}
    for map_key, name_score_list in groups.items():
        # Extract all reference image names for this map segment.
        ref_names = [n for n, _ in name_score_list]
        local_feat_path = local_feat_paths[map_key]
        # Batch load local features for all reference images in this map segment.
        local_feats = load_local_features_fn(local_feat_path, ref_names)
        frames_by_name = all_colmap_models[map_key]
        for ref_name, score in name_score_list:
            candidates_data[ref_name] = {
                "frame": frames_by_name[ref_name],        # COLMAP frame info for this image
                "local_feat": local_feats[ref_name],      # Local features for this image
                "map_key": map_key,                       # The map domain key
                "score": score                            # VPR similarity score
            }
    return candidates_data

def select_best_map_key_by_inliers(results, top_candidates):
    """
    Select the most robust map_key by voting on the total number of inliers after geometric verification.
    Falls back to VPR top-1 if no geometric-verified candidates exist.
    """
    from collections import defaultdict
    region_scores = defaultdict(int)
    for res in results:
        region_scores[res["map_key"]] += res["inliers"]
    if region_scores:
        # Choose the map_key with the highest sum of inliers
        return max(region_scores, key=region_scores.get)
    else:
        # Fallback to VPR top-1 map_key
        return top_candidates[0][0]