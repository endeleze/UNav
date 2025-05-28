import torch
import numpy as np
from unav.core.feature_filter import match_query_to_database, ransac_filter

from typing import Dict, Any, Callable, List, Tuple

def batch_local_matching_and_ransac(
    query_local_feat: Dict[str, Any],
    candidates_data: Dict[str, Dict[str, Any]],
    matcher: Callable,
    device: torch.device,
    feature_score_threshold: float = 0.09,
    min_inliers: int = 6
) -> Tuple[str, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Batch local matching + geometric verification (RANSAC) for VPR candidates.
    Returns only the best map region's 2D-3D matches for robust downstream PnP/refinement.
    
    Returns:
        best_map_key (str): The selected best map_key after geometric filtering.
        pnp_pairs (Dict): Concatenated 2D-3D pairs for best map_key.
        results (List[Dict]): Each candidate's matching info (only for best_map_key).
    """

    # 1. Local feature matching
    ref_img_names = list(candidates_data.keys())
    db_feats = [
        {
            "descriptors": torch.tensor(candidates_data[name]["local_feat"]["descriptors"], device=device),
            "keypoints": torch.tensor(candidates_data[name]["local_feat"]["keypoints"], device=device),
            "scores": torch.tensor(candidates_data[name]["local_feat"]["scores"], device=device),
            "image_size": torch.tensor(candidates_data[name]["local_feat"]["image_size"], device=device),
        }
        for name in ref_img_names
    ]
    feat1 = {
        "descriptors": torch.tensor(query_local_feat["descriptors"], device=device),
        "keypoints": torch.tensor(query_local_feat["keypoints"], device=device),
        "scores": torch.tensor(query_local_feat["scores"], device=device),
        "image_size": torch.tensor(query_local_feat["image_size"], device=device),
    }

    # 2. Batch matching and RANSAC
    names2, p0_idx, p1_idx, scores_list = match_query_to_database(
        feat1, db_feats, ref_img_names,
        local_feature_matcher=matcher,
        device=device,
        feature_score_threshold=feature_score_threshold
    )

    if not names2:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    k0_np = np.array(query_local_feat["keypoints"])
    k0_list = [k0_np] * len(names2)
    k1_list = [
        np.array(candidates_data[name]["local_feat"]["keypoints"]) for name in names2
    ]

    valid_mask, in0, in1, inlier_scores = ransac_filter(
        p0_idx, p1_idx, k0_list, k1_list, scores_list, device, threshold=min_inliers
    )

    # 3. Group by map_key for robust region selection
    grouped = dict()
    for idx, name in enumerate(names2):
        if not valid_mask[idx] or in0[idx] is None or len(in0[idx]) < min_inliers:
            continue

        candidate = candidates_data[name]
        map_key = candidate["map_key"]
        ref_frame = candidate["frame"]
        query_idxs = in0[idx]
        ref_idxs = in1[idx]

        # Only keep valid 3D-2D correspondences
        valid_pairs = [
            (qi, ri)
            for qi, ri in zip(query_idxs, ref_idxs)
            if ref_frame["points3D_xyz"][ri] is not None
        ]
        if len(valid_pairs) < min_inliers:
            continue

        query_valid_idx, ref_valid_idx = zip(*valid_pairs)
        image_points = k0_np[list(query_valid_idx)]
        object_points = np.array([ref_frame["points3D_xyz"][ri] for ri in ref_valid_idx])

        # Group by map_key
        if map_key not in grouped:
            grouped[map_key] = {
                "all_image_points": [],
                "all_object_points": [],
                "results": [],
                "total_inliers": 0,
            }
        grouped[map_key]["all_image_points"].append(image_points)
        grouped[map_key]["all_object_points"].append(object_points)
        grouped[map_key]["total_inliers"] += len(object_points)
        grouped[map_key]["results"].append({
            "ref_image_name": name,
            "map_key": map_key,
            "score": candidate["score"],
            "inliers": len(object_points),
            "query_idxs": list(query_valid_idx),
            "ref_idxs": list(ref_valid_idx),
            "object_points": object_points,
            "image_points": image_points,
            "debug": {"raw_inliers": len(query_idxs)}
        })

    if not grouped:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    # 4. Choose best_map_key by max inliers
    best_map_key = max(grouped, key=lambda k: grouped[k]["total_inliers"])
    block = grouped[best_map_key]

    if block["all_image_points"] and block["all_object_points"]:
        pnp_image_points = np.concatenate(block["all_image_points"], axis=0)
        pnp_object_points = np.concatenate(block["all_object_points"], axis=0)
    else:
        pnp_image_points = np.zeros((0, 2))
        pnp_object_points = np.zeros((0, 3))

    pnp_pairs = {
        "image_points": pnp_image_points,
        "object_points": pnp_object_points
    }
    return best_map_key, pnp_pairs, block["results"]
