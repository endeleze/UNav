import os
import time
import threading
import torch
import numpy as np

from typing import Dict, Any

# Feature/model abstraction imports
from unav.core.feature.Global_Extractors import GlobalExtractors
from unav.core.feature.local_extractor import Local_extractor

# Utility tools for I/O and matching
from unav.localizer.tools.io import load_colmap_model, load_global_features, load_local_features
from unav.localizer.tools.feature_extractor import extract_query_features
from unav.localizer.tools.retriever import (
    search_vpr_topk_candidates,
    fetch_candidates_data
)
from unav.localizer.tools.matcher import batch_local_matching_and_ransac
# MASt3R imports are lazy — only loaded when mast3r mode is active
from unav.localizer.tools.pnp import (
    refine_pose_from_queue,
    transform_pose_to_floorplan,
)

class UNavLocalizer:
    """
    UNavLocalizer: Unified Visual Place Recognition and Pose Estimation for UNav System

    - Responsible for managing all models, maps, and feature data for large-scale visual localization.
    - All heavy data loading is separated from initialization for efficiency and scalability.
    - Modular design supports multi-building, multi-floor, and multi-map environments.
    """

    def __init__(self, config):
        """
        Initialize the localizer with system configuration.
        Only sets up model pointers; heavy map/feature data are loaded on demand.

        Args:
            config: Configuration object containing all system parameters and paths.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_extractor = None
        self.global_extractor = None
        self.local_matcher = None

        # Data containers for loaded map/models/features
        self.all_colmap_models = {}      # {place__building__floor: frames_by_name}
        self.all_global_features = {}    # {place__building__floor: (features, names)}
        self.colmap_model_dirs = {}      # {key: model_dir} — for lazy COLMAP loading
        self.global_feat_paths = {}      # {place__building__floor: h5_path}
        self.local_feat_paths = {}       # {place__building__floor: h5_path}
        self.transform_matrices = {}     # {place__building__floor: np.ndarray or None}
        self._db_xy_cache = {}           # {map_key_tuple: {img_name: (x, y) floorplan px}}

        # Kick off global-feature loading (CPU/IO) in a background thread so it
        # overlaps with GPU model init below; the two phases are independent.
        self._register_map_paths()
        self._gf_thread = threading.Thread(target=self._load_all_global_features, daemon=True)
        self._gf_thread.start()
        self._init_models()

    def _init_models(self):
        """
        Initialize local and global feature extraction models (but not map/features).
        """
        feat_cfg = self.config.feature_extraction_config
        self.use_mast3r = self.config.local_feature_model == "mast3r"
        print(
            f"[INFO] Initializing models: "
            f"Local -> {self.config.local_feature_model} | "
            f"Global -> {self.config.global_descriptor_model}"
        )
        local_ext = Local_extractor(feat_cfg["local_extractor_config"])
        self.local_extractor = local_ext.extractor()
        matcher = local_ext.matcher()
        if self.use_mast3r:
            self.local_matcher = matcher  # MASt3RExtractor instance (already on device)
        else:
            self.local_matcher = matcher.to(self.device)
        self.global_extractor = GlobalExtractors(
            feat_cfg["parameters_root"],
            {self.config.global_descriptor_model: feat_cfg["global_descriptor_config"]},
            data_parallel=False
        )
        self.global_extractor.set_train(False)

    def _register_map_paths(self):
        """Register feature/model paths and load (tiny) transform matrices. Cheap."""
        for place, bld_dict in self.config.places.items():
            for building, floors in bld_dict.items():
                for floor in floors:
                    key = (place, building, floor)
                    feature_dir = os.path.join(self.config.data_final_root, place, building, floor, "features")
                    self.global_feat_paths[key] = os.path.join(feature_dir, f"global_features_{self.config.global_descriptor_model}.h5")
                    self.local_feat_paths[key] = os.path.join(feature_dir, "local_features.h5")
                    self.colmap_model_dirs[key] = os.path.join(self.config.data_final_root, place, building, floor, "colmap_map")
                    transform_path = os.path.join(self.config.data_final_root, place, building, floor, "transform_matrix.npy")
                    if os.path.exists(transform_path):
                        try:
                            self.transform_matrices[key] = np.load(transform_path)
                        except Exception as e:
                            print(f"[WARNING] Could not load transform matrix for {key}: {e}")
                            self.transform_matrices[key] = None
                    else:
                        self.transform_matrices[key] = None

    def _load_all_global_features(self):
        """Eagerly load global features for every floor (needed for VPR).

        A floor whose features do not load is invisible to retrieval: its
        queries silently land on whatever other floor looks closest, which
        for a navigation product means confidently routing a user through
        the wrong floor. Previously a missing file was skipped with no
        message at all and a failed load produced a single warning that was
        easy to miss in startup noise, so record what was skipped and end
        with an explicit inventory.
        """
        self.unavailable_floors = {}
        for key, h5_path in list(self.global_feat_paths.items()):
            if not os.path.exists(h5_path):
                self.unavailable_floors[key] = f"missing file: {h5_path}"
                print(f"[WARNING] No global features for {key}: file not found "
                      f"({h5_path}) -- this floor will be INVISIBLE to retrieval")
                continue
            try:
                feats, names = load_global_features(h5_path)
                self.all_global_features[key] = (feats, names)
                print(f"[✓] Loaded global features for {key}: {len(names)} images")
            except Exception as e:
                self.unavailable_floors[key] = f"{type(e).__name__}: {e}"
                print(f"[WARNING] Could not load global features for {key}: {e} "
                      f"-- this floor will be INVISIBLE to retrieval")
        self._report_floor_inventory()

    def _report_floor_inventory(self):
        """Print a loaded-vs-unavailable summary so a lost floor is obvious."""
        loaded = len(self.all_global_features)
        missing = getattr(self, "unavailable_floors", {}) or {}
        total = loaded + len(missing)
        if not missing:
            print(f"[INFO] Floor inventory: {loaded}/{total} floors available for retrieval.")
            return
        print(f"[ERROR] Floor inventory: {loaded}/{total} floors available -- "
              f"{len(missing)} UNAVAILABLE and invisible to retrieval:")
        for key, reason in missing.items():
            print(f"[ERROR]   {key}: {reason}")
        print("[ERROR] Queries belonging to the floors above cannot be matched "
              "and will be answered with another floor's pose. Fix the data or "
              "remove those floors from the served set.")

    def load_maps_and_features(self):
        """
        Ensure global features + transforms are loaded (COLMAP models stay lazy).
        If __init__ started a background load, just wait for it; otherwise
        (e.g. called again after maps change) do a fresh register + load.
        """
        t = getattr(self, "_gf_thread", None)
        if t is not None:
            t.join()
            self._gf_thread = None
        else:
            self._register_map_paths()
            self._load_all_global_features()
        print("[INFO] All map and feature loading complete.")

    def extract_query_features(self, query_img: np.ndarray):
        """
        Extract global and local features from the query image using the loaded models.

        Args:
            query_img (np.ndarray): Query image (H, W, 3)

        Returns:
            Tuple of (global_feature, local_feature_dict)
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
        Run visual place recognition retrieval to get top-K candidates.

        Args:
            global_feat: Query image global feature.
            top_k (int, optional): Number of top matches to return.

        Returns:
            List of (map_key, img_name, score) tuples.
        """
        topk = top_k or self.config.localization_config.get("topk", 5)
        return search_vpr_topk_candidates(
            query_feature=global_feat,
            all_map_features=self.all_global_features,
            top_k=topk,
            device=str(self.device)
        )

    def _db_floorplan_xy(self, map_key):
        """Floorplan (x, y) in px of every DB image in a map, cached. Requires the
        map's COLMAP model + transform_matrix; returns {} when no transform exists."""
        if map_key in self._db_xy_cache:
            return self._db_xy_cache[map_key]
        T = self.transform_matrices.get(map_key)
        out = {}
        if T is not None:
            model = self._ensure_colmap_model(map_key) or {}
            for name, frame in model.items():
                fp = self.transform_pose_to_floorplan(
                    frame.get("qvec"), frame.get("tvec"), T
                )
                xy = fp.get("xy") if fp else None
                if xy is not None:
                    out[name] = (float(xy[0]), float(xy[1]))
        self._db_xy_cache[map_key] = out
        return out

    def _reachability_gate(self, top_candidates, prior_xy, reach_px, keep, min_keep=3):
        """Keep VPR candidates whose DB floorplan position is within reach_px of the
        prior pose, then take the top `keep`. Falls back to the ungated top `keep`
        when too few survive (stale/unreliable prior), so matching never starves."""
        px, py = float(prior_xy[0]), float(prior_xy[1])
        r2 = float(reach_px) * float(reach_px)
        gated = []
        for cand in top_candidates:
            map_key, name = cand[0], cand[1]
            db_xy = self._db_floorplan_xy(map_key).get(name)
            if db_xy is None:
                continue
            if (db_xy[0] - px) ** 2 + (db_xy[1] - py) ** 2 <= r2:
                gated.append(cand)
        if len(gated) >= min(keep, min_keep):
            return gated[:keep]
        return top_candidates[:keep]

    def _ensure_colmap_model(self, key):
        """Lazy-load and cache a floor's COLMAP model on first use."""
        if key in self.all_colmap_models:
            return self.all_colmap_models[key]
        model_dir = self.colmap_model_dirs.get(key)
        if model_dir is None:
            return None
        t0 = time.time()
        try:
            frames_by_name = load_colmap_model(model_dir, ext=".bin")
            self.all_colmap_models[key] = frames_by_name
            print(f"[lazy] Loaded COLMAP model for {key}: {len(frames_by_name)} frames in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[WARNING] Could not lazy-load COLMAP model for {key}: {e}")
            self.all_colmap_models[key] = {}
        return self.all_colmap_models[key]

    def get_candidates_data(self, top_candidates):
        """
        Load all local features and COLMAP metadata for the VPR candidate set.

        Args:
            top_candidates: List of (map_key, img_name, score) tuples.

        Returns:
            Dict mapping image name to data needed for local matching.
        """
        for _cand in top_candidates:
            self._ensure_colmap_model(_cand[0])
        return fetch_candidates_data(
            self.all_colmap_models,
            self.local_feat_paths,
            top_candidates,
            load_local_features
        )

    def batch_local_matching_and_ransac(self, local_feat_dict, candidates_data,
                                        query_img_path=None, pp=None):
        """
        Perform local matching and geometric verification in batch.
        Dispatches to MASt3R or SuperPoint+LightGlue based on config.

        Args:
            local_feat_dict: Query local features dict (None for MASt3R).
            candidates_data: Dict of reference image data.
            query_img_path: Path to query image (required for MASt3R).

        Returns:
            best_map_key (str): Map region with most inliers.
            pnp_pairs (dict): All correspondences for pose estimation.
            results (list): Per-candidate match info.
        """
        if self.use_mast3r:
            from unav.localizer.tools.matcher import mast3r_matching_and_pnp
            mast3r_cfg = self.config.feature_extraction_config["local_extractor_config"].get("mast3r", {})
            # Perspective DB images live with the published map under the FINAL
            # tree (mapping writes them there).
            data_roots = [
                getattr(self.config, "data_final_root", None),
            ]
            return mast3r_matching_and_pnp(
                query_img_path=query_img_path,
                candidates_data=candidates_data,
                mast3r_matcher=self.local_matcher,
                colmap_models=self.all_colmap_models,
                max_nn_dist=mast3r_cfg.get("max_nn_dist", 20.0),
                # Pre-geometry correspondence count -- lives in the mast3r
                # namespace so it can never be moved by tuning the SuperPoint
                # path's post-RANSAC min_inliers (they measure different things).
                min_matches=mast3r_cfg.get("min_matches", 6),
                max_candidates=10,
                early_stop_inliers=80,
                pp=pp,
                data_roots=[r for r in data_roots if r],
            )
        else:
            return batch_local_matching_and_ransac(
                local_feat_dict,
                candidates_data,
                matcher=self.local_matcher,
                feature_score_threshold=self.config.localization_config.get("feature_score_threshold", 0.09),
                # Post-RANSAC inlier count (survivors of ransac_filter).
                min_inliers=self.config.localization_config.get("min_inliers", 50),
                device=self.device
            )

    def multi_frame_pose_refine(self, pnp_pairs, img_shape, refinement_queue, pp=None):
        """
        Multi-frame pose refinement (PnP filtering + queue update).

        Args:
            pnp_pairs: Dict of 2D-3D correspondences for this region.
            img_shape: Shape of query image.
            refinement_queue: History queue for pose refinement.

        Returns:
            Dict containing pose, queue, success, etc.
        """
        return refine_pose_from_queue(
            current_pairs=pnp_pairs,
            current_img_shape=img_shape,
            refinement_queue=refinement_queue,
            max_history=self.config.localization_config.get("max_history", 5),
            pp=pp,
        )

    def transform_pose_to_floorplan(self, qvec, tvec, transform_matrix):
        """
        Transform a 6-DoF COLMAP pose to floorplan (2D+theta) coordinates.

        Args:
            qvec (np.ndarray): Quaternion (w, x, y, z)
            tvec (np.ndarray): Translation (x, y, z)
            transform_matrix (np.ndarray): Floorplan transform

        Returns:
            Dict with position and angle on the floorplan, or None if unavailable.
        """
        return transform_pose_to_floorplan(qvec, tvec, transform_matrix)

    def localize(
        self,
        query_img: np.ndarray,
        refinement_queue: dict,
        top_k: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full end-to-end localization pipeline.

        Args:
            query_img (np.ndarray): Input image (H, W, 3)
            refinement_queue (dict): Dict tracking pose history for each map_key
            top_k (int, optional): Number of VPR candidates (default from config)

        Returns:
            Dict with keys: success, qvec, tvec, floorplan_pose, results, top_candidates,
            n_frames, refinement_queue, best_map_key, localization_time, etc.
            On failure, always includes: success=False, reason, stage, timings
        """
        # Start total localization timer
        start_time = time.time()
        timings = {}
        t0 = start_time

        # 1. Extract features from query image
        try:
            global_feat, local_feat_dict = self.extract_query_features(query_img)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during feature extraction: {e}",
                "stage": "extract_query_features",
                "timings": timings
            }
        t1 = time.time()
        timings['extract_query_features'] = t1 - t0
        t0 = t1

        # 2. VPR: retrieve top candidates
        # MASt3R is slower per-pair, use fewer VPR candidates
        effective_topk = min(top_k or 50, 10) if self.use_mast3r else top_k
        # Optional reachability prior: when the caller supplies the previous pose
        # (prior_xy, floorplan px) and a reach radius (reach_px), retrieve a larger
        # pool and keep only candidates physically reachable from the last pose,
        # suppressing look-alike corridors elsewhere. Off by default (no kwargs).
        _prior_xy = kwargs.get("prior_xy", None)
        _reach_px = kwargs.get("reach_px", None)
        _gate_on = _prior_xy is not None and _reach_px is not None
        _pool_k = max(effective_topk, 50) if _gate_on else effective_topk
        try:
            top_candidates = self.vpr_retrieve(global_feat, top_k=_pool_k)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during VPR retrieval: {e}",
                "stage": "vpr_retrieve",
                "timings": timings
            }
        t1 = time.time()
        timings['vpr_retrieve'] = t1 - t0
        t0 = t1

        if not top_candidates:
            return {
                "success": False, 
                "reason": "VPR failed (no candidates found).",
                "stage": "vpr_retrieve",
                "timings": timings
            }

        if _gate_on:
            top_candidates = self._reachability_gate(
                top_candidates, _prior_xy, _reach_px, keep=effective_topk
            )

        # 3. Gather map/model/feature data for all candidates
        try:
            candidates_data = self.get_candidates_data(top_candidates)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during candidates data gathering: {e}",
                "stage": "get_candidates_data",
                "top_candidates": top_candidates,
                "timings": timings
            }
        t1 = time.time()
        timings['get_candidates_data'] = t1 - t0
        t0 = t1

        if not candidates_data:
            return {
                "success": False,
                "reason": "No candidate data found.",
                "stage": "get_candidates_data",
                "top_candidates": top_candidates,
                "timings": timings
            }

        # 4. Local matching + RANSAC, grouped by region/map_key
        try:
            # MASt3R needs image file path — save temp file if not provided
            query_img_path = kwargs.get("query_img_path", None)
            if self.use_mast3r and query_img_path is None:
                import tempfile, cv2
                _tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(_tmp.name, query_img)
                query_img_path = _tmp.name
            pp = kwargs.get("pp", None)
            try:
                best_map_key, pnp_pairs, results = self.batch_local_matching_and_ransac(
                    local_feat_dict, candidates_data,
                    query_img_path=query_img_path,
                    pp=pp,
                )
            except torch.cuda.OutOfMemoryError:
                # Batch matching over many high-keypoint candidates can
                # exhaust GPU memory. Free the cache and retry once with the
                # top candidates by retrieval score — a slightly-less-informed
                # answer beats an exception.
                torch.cuda.empty_cache()
                _top = sorted(
                    candidates_data.items(),
                    key=lambda kv: kv[1].get("score", 0.0),
                    reverse=True,
                )[:12]
                best_map_key, pnp_pairs, results = self.batch_local_matching_and_ransac(
                    local_feat_dict, dict(_top),
                    query_img_path=query_img_path,
                    pp=pp,
                )
            # Clean up temp file
            if self.use_mast3r and 'query_img_path' not in kwargs and query_img_path:
                import os
                try: os.unlink(query_img_path)
                except: pass
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during local matching & RANSAC: {e}",
                "stage": "batch_local_matching_and_ransac",
                "top_candidates": top_candidates,
                "timings": timings
            }
        t1 = time.time()
        timings['batch_local_matching_and_ransac'] = t1 - t0
        t0 = t1

        if best_map_key is None or not results:
            return {
                "success": False,
                "reason": "No candidates passed local matching + RANSAC.",
                "stage": "batch_local_matching_and_ransac",
                "top_candidates": top_candidates,
                "timings": timings
            }

        # 5. Pose refinement
        if self.use_mast3r and "_relpose_qvec" in pnp_pairs:
            # RelPose: pose already computed, skip PnP refinement
            refine_result = {
                "success": True,
                "qvec": pnp_pairs["_relpose_qvec"],
                "tvec": pnp_pairs["_relpose_tvec"],
                "n_frames": 1,
                "new_refinement_queue": refinement_queue.get(best_map_key, {
                    "pairs": [], "initial_poses": [], "pps": []
                }),
            }
            timings['multi_frame_pose_refine'] = 0
        else:
            # PnP-based refinement (multi-frame queue)
            map_queue = refinement_queue.get(best_map_key, {
                "pairs": [], "initial_poses": [], "pps": []
            })
            try:
                refine_result = self.multi_frame_pose_refine(pnp_pairs, query_img.shape, map_queue, pp=pp)
            except Exception as e:
                return {
                    "success": False,
                    "reason": f"Exception during multi-frame pose refinement: {e}",
                    "stage": "multi_frame_pose_refine",
                    "top_candidates": top_candidates,
                    "best_map_key": best_map_key,
                    "timings": timings
                }
            t1 = time.time()
            timings['multi_frame_pose_refine'] = t1 - t0
            t0 = t1

            if not refine_result["success"]:
                return {
                    "success": False,
                    "reason": refine_result.get("reason", "Pose refinement failed."),
                    "stage": "multi_frame_pose_refine",
                    "top_candidates": top_candidates,
                    "best_map_key": best_map_key,
                    "timings": timings
                }

        # 6. Transform output pose to floorplan coordinates if possible
        colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
        transform_matrix = self.transform_matrices.get(best_map_key, None)
        try:
            floorplan_pose = (
                transform_pose_to_floorplan(colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix)
                if (colmap_pose["tvec"] is not None and transform_matrix is not None)
                else None
            )
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during pose transformation: {e}",
                "stage": "transform_pose_to_floorplan",
                "top_candidates": top_candidates,
                "best_map_key": best_map_key,
                "timings": timings
            }
        t1 = time.time()
        timings['transform_pose_to_floorplan'] = t1 - t0
        t0 = t1

        # 7. Update refinement queue for just this map region
        updated_queue = refinement_queue.copy()
        updated_queue[best_map_key] = refine_result["new_refinement_queue"]

        # 8. Output structured result
        localization_time = time.time() - start_time
        timings['total'] = localization_time

        output = {
            "success": True,
            "qvec": refine_result.get("qvec"),
            "tvec": refine_result.get("tvec"),
            "floorplan_pose": floorplan_pose,
            "results": results,
            "top_candidates": top_candidates,
            "n_frames": refine_result.get("n_frames"),
            "refinement_queue": updated_queue,
            "best_map_key": best_map_key,
            "timings": timings,
            "pnp_debug": refine_result.get("pnp_debug"),
        }
        return output