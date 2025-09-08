#!/usr/bin/env python
# coding: utf-8

# # UNav Localization Pipeline Visualization
# 
# This notebook demonstrates and visualizes the end-to-end localization pipeline for the UNav system, from feature extraction to geometric verification and final pose refinement. It is intended for inspection, debugging, and research analysis in visual indoor localization.

# ## 1. Define Data Paths and Experiment Parameters
# 
# Set up all root directories and configuration parameters required for this localization session, including place/building/floor identifiers and feature models.

# In[1]:


DATA_TEMP_ROOT = '/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data'
DATA_FINAL_ROOT = '/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data'
FEATURE_MODEL = "MixVPR"
LOCAL_FEATURE_MODEL = "superpoint+lightglue"

PLACES = {
    "Mahidol_University": {
        "ICT": ["1.1"]
    },
    "fsfasdfa":{
        "asdfa": ['df','fd'],
        "ags": ['gs','gd']
    }
}


# ## 2. Import Libraries and UNav Modules
# 
# Import all required standard libraries, configuration utilities, and matplotlib for visualization.

# In[2]:


import cv2
from unav.config import UNavConfig
import matplotlib.pyplot as plt


# ## 3. Build Configuration and Initialize the Localizer
# 
# Create the unified configuration object and instantiate the `UNavLocalizer`. Load all maps and features for the configured floors. This step may take several seconds to minutes depending on the dataset size.

# In[3]:


config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES,
    global_descriptor_model=FEATURE_MODEL,
    local_feature_model=LOCAL_FEATURE_MODEL
)
localizor_config = config.localizer_config

from unav.localizer.localizer import UNavLocalizer

localizer = UNavLocalizer(localizor_config)
localizer.load_maps_and_features()


# ## 4. Load and Display Query Image
# 
# Read a test image for localization and display it for reference.

# In[4]:


test_image_path = "/home/nattachart.tak/Data/logs/Mahidol_University/ICT/1.1/06697/images/2024-11-15_16-58-02.png"
img = cv2.imread(test_image_path)
if img is None:
    raise FileNotFoundError(f"Cannot load test image: {test_image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.title("Query Image")
plt.show()


# ## 5. Extract Global and Local Features from the Query Image
# 
# Compute the global descriptor and local keypoints for the query image. Visualize detected keypoints to inspect coverage and score distribution.

# In[5]:


from unav.visualization_tools.localization_visualization_tools import visualize_local_keypoints_on_image

global_feat, local_feat_dict = localizer.extract_query_features(img)
print("Global feature shape:", global_feat.shape)
print("Local keypoints:", local_feat_dict['keypoints'].shape)
print("Keypoint score range: {:.4f} ~ {:.4f}".format(
    local_feat_dict['scores'].min(), local_feat_dict['scores'].max()))

visualize_local_keypoints_on_image(
    img, 
    local_feat_dict['keypoints'], 
    local_feat_dict['scores'],
    figsize=(8, 8)
)


# ## 6. Visual Place Recognition (VPR): Retrieve Top-K Candidates
# 
# Use the global descriptor to retrieve the top-K most similar database images (across all mapped floors). Visualize candidate images for inspection.

# In[6]:


from unav.visualization_tools.localization_visualization_tools import plot_topk_vpr_candidates

top_candidates = localizer.vpr_retrieve(global_feat, top_k=50)
print("Top candidates:", top_candidates)

plot_topk_vpr_candidates(top_candidates, k=5, root_dir=DATA_TEMP_ROOT)


# ## 7. Load Candidate Reference Data and Visualize on Floorplans
# 
# For the VPR top-K, load their COLMAP frame information and local features. Visualize the candidate positions and headings on the registered floorplans.

# In[7]:


from unav.visualization_tools.localization_visualization_tools import visualize_candidates_on_floorplans_with_heading
from collections import Counter

candidates_data = localizer.get_candidates_data(top_candidates)
mapkey_counter = Counter([map_key for map_key, _, _ in top_candidates])
print("Candidates per floorplan:")
for map_key, count in mapkey_counter.items():
    print(f"  {map_key}: {count} candidates")

visualize_candidates_on_floorplans_with_heading(
    top_candidates=top_candidates,
    localizer=localizer,
    candidates_data=candidates_data,
    k=10,
    root_dir=DATA_FINAL_ROOT
)


# ## 8. Batch Local Matching and RANSAC Verification
# 
# Run local feature matching and geometric verification (RANSAC) for all candidates. Visualize matched keypoints between query and references to assess match quality.

# In[8]:


from unav.visualization_tools.localization_visualization_tools import visualize_query_candidate_matches
from unav.localizer.tools.io import load_local_features

best_map_key, pnp_pairs, results = localizer.batch_local_matching_and_ransac(local_feat_dict, candidates_data)
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

visualize_query_candidate_matches(
    query_img=img,
    query_kpts=local_feat_dict['keypoints'],
    results=results,
    all_candidates_kpts=all_candidates_kpts,
    root_dir=DATA_TEMP_ROOT,
    num_pairs=3,
    figsize=(12,5)
)


# ## 9. Visualize 2D-3D Crosslink on Floorplan
# 
# Project the inlier 2D-3D correspondences onto the registered floorplan and visualize the spatial relationship. This provides geometric intuition about the localization outcome.

# In[9]:


import os
import cv2
from unav.visualization_tools.localization_visualization_tools import visualize_2d_3d_crosslink

floorplan_path = os.path.join(
    DATA_FINAL_ROOT, *best_map_key, "floorplan.png"
)
floorplan_img = cv2.imread(floorplan_path)
if floorplan_img is not None and floorplan_img.ndim == 3:
    floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_BGR2RGB)

transform_matrix = localizer.transform_matrices[best_map_key]

visualize_2d_3d_crosslink(
    query_img=img,
    image_points=pnp_pairs['image_points'],
    object_points=pnp_pairs['object_points'],
    transform_matrix=transform_matrix,
    floorplan_img=floorplan_img,
    num_matches=30,
    crop_size=2000
)


# ## 10. Multi-frame Pose Refinement
# 
# Apply sliding-window multi-frame pose refinement using the current frame and history (simulated here). Print the refined pose estimate.

# In[10]:


# Prepare refinement queue (normally accumulated over video)
refinement_queue = {best_map_key: {"pairs": [], "initial_poses": [], "pps": []}}
refine_result = localizer.multi_frame_pose_refine(
    pnp_pairs, img.shape, refinement_queue[best_map_key]
)
print("Refinement result:", refine_result)


# ## 11. Project Final Camera Pose onto Floorplan
# 
# Transform the estimated pose into floorplan coordinates and visualize it. This provides the final spatial output used for navigation.

# In[11]:


from unav.visualization_tools.localization_visualization_tools import plot_camera_on_floorplan

transform_matrix = localizer.transform_matrices.get(best_map_key)
if transform_matrix is not None and refine_result["success"]:
    colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
    floorplan_pose = localizer.transform_pose_to_floorplan(
        colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix
    )
    print("Floorplan Pose (x, y, angle):", floorplan_pose)
else:
    print("Floorplan transform not available.")

if floorplan_pose is not None and floorplan_pose["xy"] is not None:
    cam_xy = tuple(floorplan_pose["xy"])
    cam_angle = floorplan_pose["ang"]
    plot_camera_on_floorplan(
        floorplan_img,
        cam_xy,
        cam_angle,
        marker='ro',
        color='red'
    )
else:
    print("Floorplan transform not available or failed.")

