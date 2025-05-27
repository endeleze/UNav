import os
import cv2
import h5py
import torch
import numpy as np
from core.feature.Global_Extractors import GlobalExtractors
from core.feature.local_extractor import Local_extractor
from localizer.tools.io import load_colmap_model, load_global_features, load_local_features
from localizer.tools.feature_extractor import extract_query_features
from localizer.tools.retriever import search_vpr_topk_candidates, fetch_candidates_data, select_best_map_key_by_inliers
from localizer.tools.matcher import batch_local_matching_and_ransac
from localizer.tools.pnp import refine_pose_from_queue, transform_pose_to_floorplan

from typing import Dict, Any

class UNavLocalizer:
    """
    UNavLocalizer manages all models, maps, and features for unified localization.
    For large-scale datasets, map/feature loading is separated from model init.
    """

    def __init__(self, config):
        """
        Only saves configuration and initializes feature extraction models.
        Does NOT load any large map/feature data here.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extraction models
        self.local_extractor = None
        self.global_extractor = None
        self._init_models()

        # Data containers, filled by .load_maps_and_features()
        self.all_colmap_models = {}       # place__building__floor -> frames_by_name
        self.all_global_features = {}     # place__building__floor -> features array
        self.global_feat_paths = {}       # place__building__floor -> h5 file path
        self.local_feat_paths = {}        # place__building__floor -> h5 file path
        self.transform_matrices = {}      # place__building__floor -> np.ndarray or None

    def _init_models(self):
        """Load feature extraction models based on config."""
        feat_cfg = self.config.feature_extraction_config
        print(f"[INFO] Initializing models: Local -> {self.config.local_feature_model} | Global -> {self.config.global_descriptor_model}")
        self.local_extractor = Local_extractor(feat_cfg["local_extractor_config"]).extractor()
        self.local_matcher = Local_extractor(feat_cfg['local_extractor_config']).matcher().to(self.device)
        self.global_extractor = GlobalExtractors(
            feat_cfg['parameters_root'],
            {self.config.global_descriptor_model: feat_cfg["global_descriptor_config"]},
            data_parallel=False
        )
        self.global_extractor.set_train(False)

    def load_maps_and_features(self):
        """
        Load all COLMAP models and global feature files for each place/building/floor.
        Call this after construction, or when new data is added.
        """
        for place in self.config.places:
            for building in self.config.buildings:
                for floor in self.config.floors:
                    key = f"{place}__{building}__{floor}"
                    # Prepare paths
                    feature_dir = os.path.join(self.config.data_final_root, place, building, floor, "features")
                    self.global_feat_paths[key] = os.path.join(feature_dir, f"global_features_{self.config.global_descriptor_model}.h5")
                    self.local_feat_paths[key] = os.path.join(feature_dir, "local_features.h5")
                    model_dir = os.path.join(self.config.data_final_root, place, building, floor, "colmap_map")
                    transform_path = os.path.join(self.config.data_final_root, place, building, floor, "transform_matrix.npy")
                    # Load COLMAP
                    try:
                        frames_by_name = load_colmap_model(model_dir, ext=".bin")
                        self.all_colmap_models[key] = frames_by_name
                        print(f"[✓] Loaded COLMAP model for {key}: {len(frames_by_name)} frames")
                    except Exception as e:
                        print(f"[WARNING] Could not load COLMAP model for {key}: {e}")
                    # Preload global features
                    h5_path = self.global_feat_paths[key]
                    if os.path.exists(h5_path):
                        try:
                            feats, names = load_global_features(h5_path)
                            self.all_global_features[key] = (feats, names)
                            print(f"[✓] Loaded global features for {key}: {len(names)} images")
                        except Exception as e:
                            print(f"[WARNING] Could not load global features for {key}: {e}")
                    # Preload transformation matrix if present
                    if os.path.exists(transform_path):
                        try:
                            matrix = np.load(transform_path)
                            self.transform_matrices[key] = matrix
                            print(f"[✓] Loaded transform matrix for {key}: shape={matrix.shape}")
                        except Exception as e:
                            print(f"[WARNING] Could not load transform matrix for {key}: {e}")
                            self.transform_matrices[key] = None
                    else:
                        self.transform_matrices[key] = None  # Not present
                        
        print("[INFO] All map and feature loading complete.")

    def extract_query_features(self, query_img: np.ndarray):
        """
        Extract global and local features from a query image.
        """
        return extract_query_features(
            query_img,
            self.global_extractor,
            self.local_extractor,
            self.config.global_descriptor_model,
            self.device
        )

    def vpr_retrieve(self, global_feat, top_k=None):
        """
        Compute VPR retrieval using query global descriptor.
        Returns a list of top-K candidate tuples.
        """
        topk = top_k or self.config.localization_config.get("topk", 5)
        return search_vpr_topk_candidates(
            query_feature=global_feat,
            all_map_features=self.all_global_features,
            top_k=topk,
            device=str(self.device)
        )

    def get_candidates_data(self, top_candidates):
        """
        For given top_candidates, fetch COLMAP frames and local feature data.
        Returns a dictionary suitable for local matching.
        """
        return fetch_candidates_data(
            self.all_colmap_models,
            self.local_feat_paths,
            top_candidates,
            load_local_features
        )

    def batch_local_matching_and_ransac(self, local_feat_dict, candidates_data):
        """
        Perform batch local matching and RANSAC verification for all VPR candidates.
        Returns:
            best_map_key (str): Selected map segment key (based on maximal inlier matches).
            pnp_pairs (dict): {'image_points': ..., 'object_points': ...} from all matches in this map.
            results (list): List of detailed match info for all valid candidates in this map.
        """
        return batch_local_matching_and_ransac(
            local_feat_dict,
            candidates_data,
            matcher=self.local_matcher,
            feature_score_threshold=self.config.localization_config.get("feature_score_threshold", 0.09),
            min_inliers=self.config.localization_config.get("min_inliers", 50),
            device=self.device
        )

    def multi_frame_pose_refine(self, pnp_pairs, img_shape, refinement_queue):
        """
        Run multi-frame pose refinement given current frame and historical pairs.
        Returns pose and updated refinement queue.
        """
        return refine_pose_from_queue(
            current_pairs=pnp_pairs,
            current_img_shape=img_shape,
            refinement_queue=refinement_queue,
            max_history=self.config.localization_config.get("max_history", 5)
        )

    def transform_pose_to_floorplan(self, qvec, tvec, transform_matrix):
        """
        Convert COLMAP pose to floorplan (2D position + angle).
        """
        return transform_pose_to_floorplan(qvec, tvec, transform_matrix)
    
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
    
    def localize(
        self,
        query_img: np.ndarray,
        refinement_queue: dict,
        top_k: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level one-shot localization API, orchestrates all steps.
        Returns pose, debug info, and updated refinement queue.

        Args:
            query_img (np.ndarray): Query image (H, W, 3).
            refinement_queue (dict): Dict mapping map_key -> {pairs, initial_poses, pps} for pose refinement.
            top_k (int): VPR candidate size.
        Returns:
            Dict with success, pose, floorplan_pose, results, updated refinement_queue, etc.
        """

        # 1. Extract global and local features from the query image
        global_feat, local_feat_dict = self.extract_query_features(query_img)

        # 2. VPR global retrieval
        top_candidates = self.vpr_retrieve(global_feat, top_k=top_k)
        if not top_candidates:
            return {"success": False, "reason": "VPR failed (no candidates)"}

        # 3. Prepare candidate map data
        candidates_data = self.get_candidates_data(top_candidates)
        if not candidates_data:
            return {"success": False, "reason": "No candidate data found", "top_candidates": top_candidates}

        # 4. Local matching + RANSAC; force all pairs to be in the best map segment
        best_map_key, pnp_pairs, results = self.batch_local_matching_and_ransac(local_feat_dict, candidates_data)
        if best_map_key is None or not results:
            return {
                "success": False,
                "reason": "No candidates passed local matching + RANSAC.",
                "top_candidates": top_candidates,
            }

        # 5. Multi-frame pose refinement ONLY within the best_map_key segment
        # If map_key not present in queue, create empty
        map_queue = refinement_queue.get(best_map_key, {
            "pairs": [],
            "initial_poses": [],
            "pps": []
        })
        refine_result = self.multi_frame_pose_refine(
            pnp_pairs, query_img.shape, map_queue
        )

        # 6. Transform to floorplan if possible
        colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
        transform_matrix = self.transform_matrices.get(best_map_key, None)
        floorplan_pose = (
            transform_pose_to_floorplan(colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix)
            if (colmap_pose["tvec"] is not None and transform_matrix is not None)
            else None
        )

        # 7. Update only this map_key in the refinement queue
        updated_queue = refinement_queue.copy()
        updated_queue[best_map_key] = refine_result["new_refinement_queue"]

        output = {
            "success": refine_result["success"],
            "qvec": refine_result.get("qvec"),
            "tvec": refine_result.get("tvec"),
            "floorplan_pose": floorplan_pose,
            "results": results,
            "top_candidates": top_candidates,
            "n_frames": refine_result.get("n_frames"),
            "refinement_queue": updated_queue,
            "best_map_key": best_map_key
        }
        if not refine_result["success"]:
            output["reason"] = refine_result.get("reason", "Pose refinement failed.")
        return output
