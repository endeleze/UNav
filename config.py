import os
import cv2
import yaml
from typing import Dict, List, Any

class UNavConfig:
    """
    Unified configuration container for the entire UNav system.
    Manages mapping, localization, and navigation module configurations in a centralized manner.
    """
    def __init__(
        self,
        data_temp_root: str = "/mnt/data/UNav-IO/temp",
        data_final_root: str = "/mnt/data/UNav-IO/final",
        places: List[str] = ["New_York_City"],
        buildings: List[str] = ["LightHouse"],
        floors: List[str] = ["3_floor", "4_floor", "6_floor"],
        mapping_place: str = "New_York_City",
        mapping_building: str = "LightHouse",
        mapping_floor: str = "3_floor",
        global_descriptor_model: str = "DinoV2Salad",
        local_feature_model: str = "superpoint+lightglue"
    ) -> None:
        """
        Initialize the unified configuration for the entire UNav system.

        Args:
            data_temp_root (str): Temporary data root directory.
            data_final_root (str): Final data root directory.
            places (List[str]): List of all supported places (campuses, cities).
            buildings (List[str]): List of all supported buildings.
            floors (List[str]): List of all supported floors.
            mapping_place (str): The primary place used for mapping.
            mapping_building (str): The primary building used for mapping.
            mapping_floor (str): The primary floor used for mapping.
            global_descriptor_model (str): Global descriptor model name.
            local_feature_model (str): Local feature extractor name.
        """
        # Data directory for mapping pipeline (single mapping_place, mapping_building, mapping_floor)
        self.data_final_dir: str = os.path.join(
            data_final_root, mapping_place, mapping_building, mapping_floor
        )

        # Prepare building_jsons for all places/buildings/floors for navigation
        building_jsons: Dict[str, Dict[str, Dict[str, str]]] = {
            place: {
                building: {
                    floor: os.path.join(data_final_root, place, building, floor, "boundaries.json")
                    for floor in floors
                }
                for building in buildings
            }
            for place in places
        }

        scale_file = os.path.join(data_final_root, "scale.json")

        # Mapping config uses only the primary mapping place/building/floor
        self.mapping_config = UNavMappingConfig(
            data_temp_root=data_temp_root,
            data_final_root=data_final_root,
            place=mapping_place,
            building=mapping_building,
            floor=mapping_floor,
            global_descriptor_model=global_descriptor_model,
            local_feature_model=local_feature_model
        )

        # Placeholder for future localization module configuration
        self.localizer_config = None

        # Navigation config uses all available places/buildings/floors and scale_file
        self.navigator_config = UNavNavigationConfig(
            building_jsons=building_jsons,
            scale_file=scale_file
        )

    def to_dict(self):
        """
        Recursively export the entire configuration as a nested dictionary.

        Returns:
            dict: A nested dictionary representing all configuration blocks.
        """
        result = {"mapping_config": self.mapping_config.to_dict()}
        if self.localizer_config is not None:
            result["localizer_config"] = self.localizer_config.to_dict()
        if self.navigator_config is not None:
            result["navigator_config"] = self.navigator_config.to_dict()
        return result

class UNavMappingConfig:
    """
    Centralized, unified configuration class for the UNav Mapping pipeline.
    Supports flexible model selection with automatic model-specific configs.
    All data, model, and output paths are managed in a unified way.
    """

    # ----------- Supported Model/Extractor Dictionaries -----------
    SUPPORTED_GLOBAL_MODELS: Dict[str, Dict[str, Any]] = {
        "MixVPR": {
            "ckpt_path": 'parameters/MixVPR/ckpts/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt',
            "pt_img_size": [320, 320],
            "cuda": True,
            "model_resize": [320, 320]
        },
        "CricaVPR": {
            "ckpt_path": 'parameters/CricaVPR/ckpts/CricaVPR_clean.pth',
            "cuda": True,
            "model_resize": [320, 320]
        },
        "DinoV2Salad": {
            "ckpt_path": 'parameters/DinoV2Salad/ckpts/dino_salad.ckpt',
            "max_image_size": 1024,
            "num_channels": 384,
            "cuda": True,
            "model_resize": [224, 224]
        },
        "NetVlad": {
            "ckpt_path": 'parameters/netvlad/paper/checkpoints',
            "arch": 'vgg16',
            "num_clusters": 64,
            "pooling": 'netvlad',
            "vladv2": False,
            "nocuda": False,
            "model_resize": [480, 640]
        },
        "AnyLoc": {
            "model_type": 'dinov2_vitg14',
            "ckpt_path": 'None',
            "max_image_size": 1024,
            "desc_layer": 31,
            "desc_facet": 'value',
            "num_clusters": 32,
            "domain": 'indoor',
            "cache_dir": 'parameters/AnyLoc/demo/cache',
            "cuda": True,
            "model_resize": [224, 224]
        }
    }

    SUPPORTED_LOCAL_EXTRACTORS: Dict[str, Dict[str, Any]] = {
        "superpoint+lightglue": {
            "detector_name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
            "matcher_name": "lightglue",
            "match_conf": {
                "width_confidence": -1,
                "depth_confidence": -1
            }
        }
    }

    # ----------- Initialization -----------
    def __init__(
        self,
        data_temp_root: str = "/mnt/data/UNav-IO/temp",
        data_final_root: str = "/mnt/data/UNav-IO/final",
        place: str = "New_York_City",
        building: str = "LightHouse",
        floor: str = "3_floor",
        global_descriptor_model: str = "DinoV2Salad",
        local_feature_model: str = "superpoint+lightglue"
    ) -> None:
        """
        Initialize the UNav mapping config.
        """
        # Root paths for temp/final data
        self.data_temp_root: str = data_temp_root
        self.data_final_root: str = data_final_root

        # Task-specific identifiers
        self.place: str = place
        self.building: str = building
        self.floor: str = floor

        # Model selection
        self.global_descriptor_model: str = global_descriptor_model
        self.local_feature_model: str = local_feature_model

        # Auto-generated subdirectories
        self.data_temp_dir: str = os.path.join(data_temp_root, place, building, floor)
        self.data_final_dir: str = os.path.join(data_final_root, place, building, floor)

        # Config blocks
        self.slam_config: Dict[str, Any] = self._init_slam_config()
        self.aligner_config: Dict[str, Any] = self._init_aligner_config()
        self.slicer_config: Dict[str, Any] = self._init_slicing_config()
        self.feature_extraction_config: Dict[str, Any] = self._init_feature_extraction_config()
        self.matcher_config: Dict[str, Any] = self._init_matching_config()
        self.colmap_config: Dict[str, Any] = self._init_colmap_config()

        # YAML config for SLAM
        self._generate_stella_vslam_yaml()

    # --------------------- Model Expansion Hints ---------------------
    # To add a new global descriptor or local extractor, simply update
    # SUPPORTED_GLOBAL_MODELS or SUPPORTED_LOCAL_EXTRACTORS with the new entry.
    # All downstream configs will pick it up automatically by name.

    # --------------------- Utility Functions ---------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Export the config as a nested dictionary for serialization or logging.
        """
        return {
            "data_temp_root": self.data_temp_root,
            "data_final_root": self.data_final_root,
            "place": self.place,
            "building": self.building,
            "floor": self.floor,
            "global_descriptor_model": self.global_descriptor_model,
            "local_feature_model": self.local_feature_model,
            "slam_config": self.slam_config,
            "aligner_config": self.aligner_config,
            "slicer_config": self.slicer_config,
            "feature_extraction_config": self.feature_extraction_config,
            "matcher_config": self.matcher_config,
            "colmap_config": self.colmap_config
        }

    def __repr__(self) -> str:
        """
        Human-readable summary for debugging.
        """
        return (f"<UNavMappingConfig {self.place}/{self.building}/{self.floor} "
                f"G: {self.global_descriptor_model} L: {self.local_feature_model}>")

    # ----------- SLAM Config -----------
    def _init_slam_config(self) -> dict:
        """
        Initialize config for stella_vslam_dense. Container paths and host paths both managed.
        """
        container_data_root = "/data"
        host_data_root = self.data_temp_root

        output_base = os.path.join(container_data_root, self.place, self.building, self.floor, "stella_vslam_dense")
        return {
            "container_name": f"vslam_{self.floor}",
            "gpu_id": 0,
            "viewer": False,
            "vocab_path": os.path.join(container_data_root, "orb_vocab.fbow"),
            "config_yaml": os.path.join(container_data_root, "equirectangular.yaml"),
            "video_path": os.path.join(container_data_root, self.place, self.building, self.floor, f"{self.floor}.mp4"),
            "output_dir": output_base,
            "eval_log_dir": os.path.join(output_base, "eval_logs"),
            "map_db_out": os.path.join(output_base, "final_map.msg"),
            "pc_out": os.path.join(output_base, "output_cloud.ply"),
            "kf_out": os.path.join(output_base, "keyframes"),
            "host_data_root": host_data_root,
            "container_data_root": container_data_root,
            "host_eval_log_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "eval_logs"),
            "host_keyframe_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "keyframes")
        }

    # ----------- Aligner Config -----------
    def _init_aligner_config(self) -> dict:
        """
        Config for aligning SLAM point cloud and floorplan.
        """
        temp_dir = os.path.join(self.data_temp_dir, "aligner")
        return {
            "temp_dir": temp_dir,
            "final_dir": self.data_final_dir,
            "scale_file": os.path.join(self.data_final_root, 'scale.json'),
            "map_db_out": os.path.join(self.data_temp_dir, "stella_vslam_dense", "final_map.msg"),
            "floorplan_path": os.path.join(self.data_final_dir, 'floorplan.png')
        }

    # ----------- Slicing (Perspective Image) Config -----------
    def _init_slicing_config(self) -> dict:
        """
        Config for slicing equirectangular keyframes into perspective images.
        """
        return {
            "input_keyframe_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "keyframes"),
            "trajectory_file": os.path.join(self.data_temp_dir, "stella_vslam_dense", "eval_logs", "keyframe_trajectory.txt"),
            "output_perspective_dir": os.path.join(self.data_temp_dir, "perspectives"),
            "rotate_along_local_y_axis": False,
            "num_perspectives": 18,
            "fov": 90,
            "pitch": 0
        }

    # ----------- Feature Extraction Config -----------
    def _init_feature_extraction_config(self) -> dict:
        """
        Config for local and global feature extraction, including all model configs.
        """
        if self.global_descriptor_model not in self.SUPPORTED_GLOBAL_MODELS:
            raise ValueError(f"Unsupported global descriptor model: {self.global_descriptor_model}")
        if self.local_feature_model not in self.SUPPORTED_LOCAL_EXTRACTORS:
            raise ValueError(f"Unsupported local feature extractor: {self.local_feature_model}")

        feature_dir = os.path.join(self.data_final_dir, "features")

        return {
            "parameters_root": self.data_temp_root,
            "input_perspective_dir": os.path.join(self.data_temp_dir, "perspectives"),
            "output_feature_dir": feature_dir,
            "local_feature_model": self.local_feature_model,
            "local_extractor_config": {
                self.local_feature_model: self.SUPPORTED_LOCAL_EXTRACTORS[self.local_feature_model]
            },
            "global_descriptor_model": self.global_descriptor_model,
            "global_descriptor_config": self.SUPPORTED_GLOBAL_MODELS[self.global_descriptor_model],
            "local_feat_save_path": os.path.join(feature_dir, "local_features.h5"),
            "global_feat_save_path": os.path.join(feature_dir, f"global_features_{self.global_descriptor_model}.h5")
        }

    # ----------- Feature Matching Config -----------
    def _init_matching_config(self) -> dict:
        """
        Config for feature matching and geometric verification.
        """
        feature_dir = os.path.join(self.data_final_dir, "features")
        sfm_dir = os.path.join(self.data_temp_dir, "colmap_sfm")
        return {
            "feature_dir": feature_dir,
            "colmap_local_feature_file": os.path.join(sfm_dir, "local_features.txt"),
            "colmap_match_file": os.path.join(sfm_dir, "matches.txt"),
            "top_k_matches": 50,
            "min_keypoints": 50,
            "feature_score_threshold": 0.09,
            "gv_threshold_pos": 0.005,
            "gv_threshold_angle_deg": 10.0,
        }

    # ----------- COLMAP Config -----------
    def _init_colmap_config(self) -> dict:
        """
        Config for COLMAP triangulation, input/output paths.
        """
        colmap_read_root = os.path.join(self.data_temp_dir, "colmap_sfm")
        sparse_dir = os.path.join(colmap_read_root, "sparse", "0")
        colmap_output_dir = os.path.join(self.data_final_dir, "colmap_map")
        return {
            "sparse_dir": sparse_dir,
            "colmap_output_dir": colmap_output_dir,
            "database_path": os.path.join(colmap_read_root, "database.db"),
            "camera_file": os.path.join(sparse_dir, "cameras.txt"),
            "image_file": os.path.join(sparse_dir, "images.txt"),
            "pairs_txt": os.path.join(colmap_read_root, "pairs.txt"),
            "match_file": os.path.join(colmap_read_root, "matches.h5"),
        }

    # ----------- YAML Generation (for SLAM) -----------
    def _generate_stella_vslam_yaml(self) -> None:
        """
        Automatically generate equirectangular.yaml for stella_vslam_dense using video metadata.
        """
        video_path = os.path.join(self.data_temp_root, self.place, self.building, self.floor, f"{self.floor}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        yaml_content = {
            'Camera': {
                'name': 'Insta360',
                'setup': 'monocular',
                'model': 'equirectangular',
                'fps': round(fps, 2),
                'cols': cols,
                'rows': rows,
                'color_order': 'BGR'
            },
            'Preprocessing': {
                'min_size': 800,
                'mask_rectangles': [
                    [0.0, 1.0, 0.0, 0.1],
                    [0.0, 1.0, 0.84, 1.0],
                    [0.0, 0.2, 0.7, 1.0],
                    [0.8, 1.0, 0.7, 1.0]
                ]
            },
            'Feature': {
                'name': 'default ORB feature extraction setting',
                'scale_factor': 1.2,
                'num_levels': 8,
                'ini_fast_threshold': 20,
                'min_fast_threshold': 7
            },
            'Mapping': {
                'keyframe_insert_interval': 7,
                'baseline_dist_thr_ratio': 0.02,
                'redundant_obs_ratio_thr': 0.95,
            },
            'LoopDetector': {
                'enabled': True,
                'reject_by_graph_distance': True,
                'min_distance_on_graph': 50
            },
            'SocketPublisher': {
                'image_quality': 80
            },
            'PatchMatch': {
                'enabled': True,
                'cols': 640,
                'rows': 320,
                'min_patch_std_dev': 0,
                'patch_size': 7,
                'patchmatch_iterations': 4,
                'min_score': 0.1,
                'min_consistent_views': 3,
                'depthmap_queue_size': 5,
                'depthmap_same_depth_threshold': 0.08,
                'min_views': 1,
                'pointcloud_queue_size': 4,
                'pointcloud_same_depth_threshold': 0.08,
                'min_stereo_score': 0
            }
        }

        yaml_output_path = os.path.join(self.data_temp_root, "equirectangular.yaml")
        with open(yaml_output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"[✓] YAML written to: {yaml_output_path}.")

class UNavLocalizationConfig:
    pass

class UNavNavigationConfig:
    def __init__(self, building_jsons: Dict[str, Dict[str, str]], scale_file: str = None):
        self.building_jsons = building_jsons    # Dict[str, Dict[str, str]]
        self.scale_file = scale_file            # str or None