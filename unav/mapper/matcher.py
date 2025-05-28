import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

from unav.config import UNavMappingConfig
from unav.core.feature.local_extractor import Local_extractor
from unav.core.colmap.utils_pose import load_colmap_images_file
from unav.core.feature_filter import match_query_to_database, ransac_filter
from unav.mapper.tools.matcher.geometry import fast_find_adjacent_and_pose_pairs

def compute_similarity_and_generate_topk_pairs(
    descriptor_h5_path: str,
    top_k: int = 50,
    batch_size: int = 500
) -> List[Tuple[str, str]]:
    """
    Compute Top-K most similar image pairs based on global descriptor similarity.

    Args:
        descriptor_h5_path (str): Path to the HDF5 file containing global descriptors.
        top_k (int): Number of top matches per image.
        batch_size (int): Batch size for similarity computation.

    Returns:
        List[Tuple[str, str]]: List of (img1, img2) candidate pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(descriptor_h5_path, 'r') as f:
        img_names = list(f.keys())
        descriptors = torch.stack([torch.tensor(f[name][:]) for name in img_names], dim=0).to(device)
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)

    N = descriptors.shape[0]
    sim_matrix = torch.full((N, N), -float('inf'), device='cpu')

    # Compute similarity in batches for efficiency
    for i in range(0, N, batch_size):
        desc_batch = descriptors[i:i+batch_size]
        sim_batch = torch.einsum('bd,nd->bn', desc_batch, descriptors).cpu()
        for j in range(sim_batch.shape[0]):
            sim_batch[j, i + j] = -float('inf')  # Prevent self-matching
        sim_matrix[i:i+batch_size] = sim_batch

    pairs = [
        (img_names[i], img_names[j])
        for i in range(N)
        for j in sim_matrix[i].numpy().argsort()[::-1][:top_k]
    ]
    return pairs

def generate_and_stream_colmap(
    config: UNavMappingConfig
) -> List[Tuple[str, str]]:
    """
    Full pipeline: 
    1. Generate candidate pairs (adjacency + global retrieval).
    2. Perform local feature matching with LightGlue.
    3. Apply RANSAC for geometric verification.
    4. Stream output to COLMAP pairs.txt and matches.h5 format.

    Args:
        config (UNavMappingConfig): Project config object.

    Returns:
        List[Tuple[str, str]]: List of all verified image pairs.
    """
    slicer_config = config.slicer_config
    feat_cfg = config.feature_extraction_config
    matcher_config = config.matcher_config
    colmap_config = config.colmap_config
    
    pairs_txt = colmap_config['pairs_txt']
    matches_h5 = colmap_config['match_file']
    image_file = colmap_config["image_file"]
    
    # Load poses and collect image names
    poses = load_colmap_images_file(image_file)
    with h5py.File(feat_cfg['local_feat_save_path'], 'r') as f:
        img_names = list(f.keys())

    # Step 1: Generate candidate pairs (adjacency + global retrieval)
    direct_pairs = fast_find_adjacent_and_pose_pairs(
        img_names,
        poses,
        gv_threshold_pos=matcher_config["gv_threshold_pos"],
        gv_threshold_angle_deg=matcher_config["gv_threshold_angle_deg"],
        gv_fov_deg=slicer_config["fov"]
    )
    retrieval_pairs = compute_similarity_and_generate_topk_pairs(
        feat_cfg['global_feat_save_path'], 
        top_k=matcher_config["top_k_matches"]
    )
    all_pairs = list({(min(a, b), max(a, b)) for a, b in set(direct_pairs) | set(retrieval_pairs)})

    # Step 2: Group pairs by img1 for batch processing
    img1_groups: Dict[str, List[str]] = {}
    for a, b in all_pairs:
        img1_groups.setdefault(a, []).append(b)

    # Step 3: Prepare output files
    os.makedirs(os.path.dirname(pairs_txt), exist_ok=True)
    os.makedirs(os.path.dirname(matches_h5), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = Local_extractor(
        feat_cfg['local_extractor_config']
    ).matcher().to(device)

    verified_pairs: List[Tuple[str, str]] = []
    
    # Step 4: Main matching and verification loop
    with h5py.File(feat_cfg['local_feat_save_path'], 'r') as f_local, \
         h5py.File(matches_h5, 'w') as f_matches, \
         open(pairs_txt, 'w') as f_pairs:
        
        skipped = 0
        progress = tqdm(img1_groups.items(), desc="Matching & Verifying")
        
        for img1, img2_list in progress:
            # Load img1 local features
            grp1 = f_local[img1]
            feat1 = {
                'descriptors': torch.tensor(grp1['descriptors'][:], device=device),
                'keypoints': torch.tensor(grp1['keypoints'][:], device=device),
                'scores': torch.tensor(grp1['scores'][:], device=device),
                'image_size': torch.tensor(grp1['image_size'][:], device=device)
            }
            # Load img2 features in batch
            db_feats: List[dict] = []
            for img2 in img2_list:
                grp2 = f_local[img2]
                db_feats.append({
                    'descriptors': torch.tensor(grp2['descriptors'][:], device=device),
                    'keypoints': torch.tensor(grp2['keypoints'][:], device=device),
                    'scores': torch.tensor(grp2['scores'][:], device=device),
                    'image_size': torch.tensor(grp2['image_size'][:], device=device)
                })

            # Local feature matching (LightGlue/SuperGlue etc.)
            names2, p0_idx, p1_idx, scores_list = match_query_to_database(
                feat1, db_feats, img2_list, matcher, device,
                feature_score_threshold=matcher_config["feature_score_threshold"],
                threshold=matcher_config["min_keypoints"]
            )
            if not names2:
                skipped += 1
                progress.set_postfix(skipped=skipped)
                continue

            # RANSAC verification and inlier extraction
            k0_np = feat1['keypoints'].cpu().numpy()
            k0_list = [k0_np] * len(names2)
            k1_list = [f_local[name]['keypoints'][:] for name in names2]
            valid_mask, in0, in1, inlier_scores = ransac_filter(
                p0_idx, p1_idx, k0_list, k1_list, scores_list, device,
                threshold=matcher_config["min_keypoints"]
            )

            # Stream writing of verified matches and pairs
            for img2, flag, idx0, idx1, scores in zip(names2, valid_mask, in0, in1, inlier_scores):
                if not flag:
                    continue

                group = f_matches.create_group(f"{img1}_{img2}")
                group.create_dataset('matches0', data=np.stack([idx0, idx1], axis=1))
                group.create_dataset('matching_scores0', data=scores)
                f_pairs.write(f"{img1} {img2}\n")

                verified_pairs.append((img1, img2))
    return verified_pairs
