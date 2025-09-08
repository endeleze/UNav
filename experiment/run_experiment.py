import argparse
import cv2
import sys, os
from os.path import dirname,realpath,abspath
import pandas as pd
import time as time_exp_use_case
import numpy as np
from pathlib import Path

# Get the absolute path of the current file's directory
current_dir = dirname(abspath(__file__))

# Get the parent directory's path
parent_dir = dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from unav.localizer.localizer import UNavLocalizer
from unav.localizer.tools.io import load_local_features
from unav.config import UNavConfig
import experiment.experiment_config as experiment_config
import experiment.distance_error as distance_error


def get_image(test_image_path):
    img = cv2.imread(test_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load test image: {test_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_pose(localizer, best_map_key, refine_result):
    transform_matrix = localizer.transform_matrices.get(best_map_key)
    cam_xy = (None, None)
    cam_angle = None
    if transform_matrix is not None and refine_result["success"]:
        colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
        floorplan_pose = localizer.transform_pose_to_floorplan(
            colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix
        )
        print("Floorplan Pose (x, y, angle):", floorplan_pose)

        cam_xy = tuple(floorplan_pose["xy"])
        cam_angle = floorplan_pose["ang"]
    elif not refine_result["success"]:
        print(refine_result["reason"])
        
    return *cam_xy, cam_angle

def __get_batch_local_matching_and_ransac(localizer, local_feat_dict, candidates_data):
    """
    Ransac fails on some query images in a nondeterministic manner. 
    Function wraps 3 attempts of the RANSAC call, and gives up if not found.
    Returns correct result or raises ValueError exception.
    """
    counter = 0
    max_counts = 3
    while counter < max_counts:
        try:
            r = localizer.batch_local_matching_and_ransac(local_feat_dict, candidates_data)
            break
        except Exception as e:
            counter+=1
            if counter == max_counts:
                raise ValueError(f"Ransac failed to converge on a (stochastic) consensus after {max_counts} attempts with error {e}")
    return r or None

def get_best_map_key(localizer, local_feat_dict, candidates_data):
    
    best_map_key, pnp_pairs, results = __get_batch_local_matching_and_ransac(localizer, local_feat_dict, candidates_data)
    print("Best map key:", best_map_key)
    print("Number of candidates after local matching:", len(results))
    print("PnP pairs 2D shape:", pnp_pairs['image_points'].shape, "3D shape:", pnp_pairs['object_points'].shape)

    # Preload all candidate keypoints for visualization
    all_candidates_kpts = {}
    for res in results:
        map_key = res['map_key']
        ref_name = res['ref_image_name']
        if map_key not in all_candidates_kpts:
            all_candidates_kpts[map_key] = {}
        if ref_name not in all_candidates_kpts[map_key]:
            h5_path = localizer.local_feat_paths[map_key]
            feats = load_local_features(h5_path, [ref_name])
            all_candidates_kpts[map_key][ref_name] = feats[ref_name]['keypoints']
    return best_map_key, pnp_pairs

class ExperimentTime:
    @staticmethod
    def get_time_ms(): return time_exp_use_case.time_ns() // 1_000_000
    @staticmethod
    def get_time_ms_duration(s): 
        return ExperimentTime.get_time_ms() - s


def update_dict(alg:str, config_exp:dict, place='Mahidol_University', building='ICT') -> dict:
    """
    Will replace `global_descriptor_model:str` with `algorithm:str`.
    Will break if a new dictionary is added.
    """
    alg_old = config_exp['global_descriptor_model']
    for k,v in config_exp.items():
        if isinstance(v, str):
            config_exp[k] = v.replace(alg_old,alg)
        elif isinstance(v, dict):
            l = config_exp[k][place][building]
            for i,f in enumerate(l):
                l[i] = l[i].replace(alg_old,alg)
            config_exp[k][place][building] = l
        elif isinstance(v, int):
            pass
        else:
            raise ValueError(f'Unsupported value type {type(v)}')
    return config_exp

"""
Run as:
    python3 run_experiment.py -x "/path/to/exp_config.json"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default=None, )
    parser.add_argument('-x', '--experiment', type=str, default=None, help='Experiment config filepath for experimental VPR localization.')
    args = parser.parse_args()

    config_exp = experiment_config.import_config(fn=args.experiment)
    config_exp = update_dict(args.algorithm, config_exp)

    DATA_FINAL_ROOT = config_exp.get('data_final_root', "/mnt/data/UNav-IO/data")
    FEATURE_MODEL = args.algorithm #onfig_exp.get('global_descriptor_model', "DinoV2Salad")
    config_exp['global_descriptor_model'] = args.algorithm
    LOCAL_FEATURE_MODEL = config_exp.get('local_feature_model', "superpoint+lightglue")
    PLACES = config_exp.get('places')
# , {
#                 "New_York_City": {
#                     "LightHouse": ["3_floor", "4_floor", "6_floor"]
#                 }
#             }

    config = UNavConfig(
        data_final_root=DATA_FINAL_ROOT,
        places=PLACES,
        global_descriptor_model=FEATURE_MODEL,
        local_feature_model=LOCAL_FEATURE_MODEL
    )
    localizor_config = config.localizer_config
    localizer = UNavLocalizer(localizor_config)
    localizer.load_maps_and_features()


    r = []
    groundtruth_df = pd.read_csv(config_exp['ground_truth_img_list'])
    image_filepaths = groundtruth_df[config_exp['img_list_attr']].apply(lambda x : f"{config_exp['path_to_images']}/{x}.{config_exp['img_ext']}").tolist()

    #image_filepaths = ["/mnt/data/UNav-IO/test/photos/LightHouse/3-1.jpg"]
    for trial_num in range(config_exp['num_trials']):
        for image_filepath in image_filepaths:
            img = get_image(image_filepath)
            cx,cy,ang = None,None,None
            error = None
            time_start = ExperimentTime.get_time_ms()
            try:
                global_feat, local_feat_dict = localizer.extract_query_features(img)
                top_candidates = localizer.vpr_retrieve(global_feat, top_k=50)
                candidates_data = localizer.get_candidates_data(top_candidates)
                #print(candidates_data)
                best_map_key, pnp_pairs = get_best_map_key(localizer, local_feat_dict, candidates_data)
                refinement_queue = {best_map_key: {"pairs": [], "initial_poses": [], "pps": []}}
                time_start_multiframe = ExperimentTime.get_time_ms()
                refine_result = localizer.multi_frame_pose_refine(
                    pnp_pairs, img.shape, refinement_queue[best_map_key]
                )
                print(f'Multiframe time (ms): {ExperimentTime.get_time_ms_duration(time_start_multiframe)}')

                cx,cy,ang = get_pose(localizer, best_map_key, refine_result)
            except Exception as e:
                error = str(e)
            total_time = ExperimentTime.get_time_ms_duration(time_start)
            r += [
                {'coordx':cx,
                    'coordy':cy,
                    'angle':ang,
                    'time_ms':total_time,
                    'image_fn':image_filepath,
                    'trial_num':trial_num,
                    'error':error,
                }]

    root_dir = config_exp['root_dir']
    filename_parts = config_exp['results_fn'].split('.')
    results_fn = f'{".".join(filename_parts[:-1])}_{args.algorithm}.{filename_parts[-1]}'
    
    out_path = config_exp['results_out_path']
    result_file = f'{root_dir}/{out_path}/{results_fn}'
    pd.DataFrame.from_records(r).to_excel(result_file)
    print(f"==== Predicted coordinates were saved to '{result_file}'. ====")

    print("==== Calculate distance errors ====")
    distance_error_fn = distance_error.save_calculated_distance_error(config_exp, results_fn)
    print(f"==== Distance errors were saved to {distance_error_fn} ====")