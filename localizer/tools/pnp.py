import pyimplicitdist
import poselib
import numpy as np
from typing import Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R

def refine_pose_from_queue(
    current_pairs: Dict[str, np.ndarray],      # Dict with "image_points": [N,2], "object_points": [N,3]
    current_img_shape: Tuple[int, int, int],   # (H, W, C)
    refinement_queue: Dict[str, list],         # Contains "pairs", "initial_poses", "pps"
    max_history: int = 5
) -> Dict[str, Any]:
    """
    Refine camera pose using multi-frame implicit pose refinement.
    The current frame is solved with coarse pose (RANSAC), then added to a sliding window history for joint optimization.

    Args:
        current_pairs (dict): Current frame's 2D-3D matches:
            {
                "image_points": np.ndarray of shape [N, 2],
                "object_points": np.ndarray of shape [N, 3]
            }
        current_img_shape (tuple): Image shape as (H, W, C).
        refinement_queue (dict): Historical information, with keys:
            - "pairs": list of dicts as above
            - "initial_poses": list of pyimplicitdist.CameraPose
            - "pps": list of np.ndarray (principal points [2])
        max_history (int): Maximum number of frames to use in the history.

    Returns:
        dict: {
            "success": bool,
            "qvec": np.ndarray (4,),  # Refined quaternion
            "tvec": np.ndarray (3,),  # Refined translation
            "n_frames": int,          # Number of frames in optimization
            "new_refinement_queue": dict (updated history for next call)
        }
    """
    H, W = current_img_shape[:2]
    pp = np.array([W / 2, H / 2])  # Use image center as principal point

    # Require at least 6 correspondences for robust PnP
    if len(current_pairs["image_points"]) < 6:
        return {"success": False, "reason": "Not enough 2D-3D correspondences for pose estimation."}

    # 1. Coarse pose estimation for current frame (RANSAC + non-linear refinement)
    try:
        # Center the 2D points for radial pose estimation
        pts2d_centered = current_pairs["image_points"] - pp
        poselib_pose, info = poselib.estimate_1D_radial_absolute_pose(
            pts2d_centered, current_pairs["object_points"], {"max_reproj_error": 6.0}
        )
        # Filter inliers
        p2d_inlier = current_pairs["image_points"][info["inliers"]]
        p3d_inlier = current_pairs["object_points"][info["inliers"]]

        # Build initial CameraPose object
        initial_pose = pyimplicitdist.CameraPose()
        initial_pose.q_vec = poselib_pose.q
        initial_pose.t = poselib_pose.t

        # Refine pose for current frame (nonlinear optimization)
        refine_opt = pyimplicitdist.PoseRefinement1DRadialOptions()
        refined = pyimplicitdist.pose_refinement_1D_radial(
            p2d_inlier, p3d_inlier, initial_pose, pp, refine_opt
        )
        refined_pose = refined["pose"]  # CameraPose object
        refined_pp = refined["pp"]      # Refined principal point
    except Exception as e:
        return {"success": False, "reason": f"Coarse pose estimation failed: {e}"}

    # 2. Update history (sliding window of frames)
    pairs_new = refinement_queue.get("pairs", []).copy()
    initial_poses_new = refinement_queue.get("initial_poses", []).copy()
    pps_new = refinement_queue.get("pps", []).copy()

    pairs_new.append({"image_points": p2d_inlier, "object_points": p3d_inlier})
    initial_poses_new.append(refined_pose)
    pps_new.append(refined_pp)
    # Keep only the latest max_history frames
    if len(pairs_new) > max_history:
        pairs_new = pairs_new[-max_history:]
        initial_poses_new = initial_poses_new[-max_history:]
        pps_new = pps_new[-max_history:]

    # 3. Multi-frame pose refinement (all frames jointly)
    list_2d = [item["image_points"] for item in pairs_new]
    list_3d = [item["object_points"] for item in pairs_new]
    mean_pp = np.mean(pps_new, axis=0)
    cm_opt = pyimplicitdist.CostMatrixOptions()
    refinement_opt = pyimplicitdist.PoseRefinementOptions()
    cost_matrix = pyimplicitdist.build_cost_matrix_multi(list_2d, cm_opt, mean_pp)
    refined_poses = pyimplicitdist.pose_refinement_multi(
        list_2d, list_3d, cost_matrix, mean_pp, initial_poses_new, refinement_opt
    )

    # The last pose corresponds to the current frame
    pose_obj = refined_poses[-1]
    qvec, tvec = pose_obj.q_vec, pose_obj.t

    return {
        "success": True,
        "qvec": qvec,
        "tvec": tvec,
        "n_frames": len(list_2d),
        "new_refinement_queue": {
            "pairs": pairs_new,
            "initial_poses": initial_poses_new,
            "pps": pps_new
        }
    }

def colmap2world(tvec, qvec):
    """
    Convert COLMAP tvec/qvec to camera center in world coordinate.
    Args:
        tvec (np.ndarray): COLMAP tvec (3,)
        qvec (np.ndarray): COLMAP quaternion (4,) [w, x, y, z]
    Returns:
        cam_center (np.ndarray): (3,)
        heading_rotvec (np.ndarray): (3,) rotation vector (not used in 2D)
    """
    # COLMAP quaternion is [w, x, y, z]; scipy expects [x, y, z, w]
    quat_xyzw = [qvec[1], qvec[2], qvec[3], qvec[0]]
    r = R.from_quat(quat_xyzw)
    rmat = r.as_matrix()
    cam_center = -rmat.T @ tvec
    r_world = R.from_matrix(rmat.T)
    return cam_center, r_world

def transform_pose_to_floorplan(
    qvec: np.ndarray,
    tvec: np.ndarray,
    transform_matrix: np.ndarray
) -> dict:
    """
    Project the COLMAP camera pose onto the floorplan and compute the heading angle,
    taking into account the 2D affine transformation.

    Args:
        qvec (np.ndarray): COLMAP quaternion [w, x, y, z], shape (4,)
        tvec (np.ndarray): COLMAP translation vector, shape (3,)
        transform_matrix (np.ndarray): Floorplan 2D affine transform, shape (2, 4)

    Returns:
        dict: {
            "xy": np.ndarray [2,],   # (x, y) in floorplan coordinates
            "ang": float             # heading angle in degrees [0, 360)
        }
    """
    if qvec is None or tvec is None or transform_matrix is None:
        return {"xy": None, "ang": None}

    # Step 1: Convert to world coordinate system
    cam_center, r_world = colmap2world(tvec, qvec)

    # Step 2: Project camera center onto the floorplan
    xyz1 = np.append(cam_center, 1.0)       # Shape: (4,)
    xy_fp = transform_matrix @ xyz1          # Shape: (2,)

    # Step 3: Compute camera forward vector in world coordinates (COLMAP uses +Z)
    cam_forward = r_world.apply(np.array([0, 0, 1]))  # Forward in world, shape: (3,)

    # Step 4: Project a forward point onto the floorplan (e.g., 1m in front)
    forward_xyz = cam_center + cam_forward
    forward_xyz1 = np.append(forward_xyz, 1.0)
    xy_fp_fwd = transform_matrix @ forward_xyz1

    # Step 5: Compute heading angle in the floorplan
    vec = xy_fp_fwd - xy_fp
    ang = np.degrees(np.arctan2(vec[1], vec[0])) % 360

    return {
        "xy": xy_fp,
        "ang": ang
    }
