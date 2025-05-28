import numpy as np
import torch
from typing import Dict, Tuple, List, Any, Callable

def search_vpr_topk_candidates(
    query_feature: np.ndarray,
    all_map_features: Dict[Tuple[str, str, str], Tuple[np.ndarray, List[str]]],
    top_k: int = 5,
    device: str = "cuda"
) -> List[Tuple[Tuple[str, str, str], str, float]]:
    """
    Search the Top-K most similar reference images for a query image across all map regions.

    Args:
        query_feature (np.ndarray): Query global descriptor, shape [D].
        all_map_features (Dict[Tuple[str, str, str], Tuple[np.ndarray, List[str]]]):
            Dictionary where each key is a (place, building, floor) tuple,
            and value is a tuple of (features [N, D], image_names [N]).
        top_k (int): Number of overall Top-K matches to return.
        device (str): Torch device string, e.g., "cuda" or "cpu".

    Returns:
        List[Tuple[Tuple[str, str, str], str, float]]:
            Each tuple is ((place, building, floor), ref_image_name, similarity_score),
            sorted descending by score.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # Normalize query feature
    qfeat = torch.tensor(query_feature, dtype=torch.float32, device=device)
    qfeat = torch.nn.functional.normalize(qfeat, dim=0)

    candidates = []
    # For each map region, compute similarities and collect candidates
    for map_key, (features, names) in all_map_features.items():
        feats = torch.tensor(features, dtype=torch.float32, device=device)
        feats = torch.nn.functional.normalize(feats, dim=1)
        sims = feats @ qfeat  # [N]
        top_scores, top_indices = torch.topk(sims, k=min(top_k, feats.shape[0]))
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            candidates.append((map_key, names[idx], score))

    # Global Top-K sort
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]


def fetch_candidates_data(
    all_colmap_models: Dict[Tuple[str, str, str], Dict[str, Any]],
    local_feat_paths: Dict[Tuple[str, str, str], str],
    top_candidates: List[Tuple[Tuple[str, str, str], str, float]],
    load_local_features_fn: Callable[[str, List[str]], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Batch fetches COLMAP frames and local features for a list of VPR retrieval candidates.

    Args:
        all_colmap_models (Dict[Tuple[str, str, str], Dict[str, Any]]): 
            Mapping from (place, building, floor) tuple to {image_name: frame_info}.
        local_feat_paths (Dict[Tuple[str, str, str], str]): 
            Mapping from (place, building, floor) tuple to local feature HDF5 file path.
        top_candidates (List[Tuple[Tuple[str, str, str], str, float]]): 
            Each tuple is ((place, building, floor), ref_img_name, score) from retrieval.
        load_local_features_fn (Callable[[str, List[str]], Dict[str, Any]]): 
            Function to batch load local features given HDF5 file path and list of image names.

    Returns:
        Dict[str, Dict[str, Any]]:
            {
                ref_image_name: {
                    "frame":      COLMAP frame dict,
                    "local_feat": local feature dict,
                    "map_key":    (place, building, floor) tuple,
                    "score":      similarity score (float)
                }
            }
    """
    # Group by map_key for efficient loading
    groups = {}
    for map_key, ref_name, score in top_candidates:
        groups.setdefault(map_key, []).append((ref_name, score))

    candidates_data = {}
    for map_key, name_score_list in groups.items():
        ref_names = [n for n, _ in name_score_list]
        local_feat_path = local_feat_paths[map_key]
        # Batch load all local features for this region
        local_feats = load_local_features_fn(local_feat_path, ref_names)
        frames_by_name = all_colmap_models[map_key]
        for ref_name, score in name_score_list:
            candidates_data[ref_name] = {
                "frame":      frames_by_name[ref_name],
                "local_feat": local_feats[ref_name],
                "map_key":    map_key,
                "score":      score
            }
    return candidates_data
