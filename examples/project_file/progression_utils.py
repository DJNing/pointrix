import torch
import cv2
import numpy as np
from pointrix.utils.pose import ConcatRT, quat_to_rotmat, apply_quaternion

def retrieve_point_cloud(depth: torch.Tensor, K: torch.Tensor, ext: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Retrieve 3D points given a depth map and camera intrinsics (K), and extrinsics. When extrinsics are None, use the identity matrix.

    Args:
        depth (torch.Tensor): Depth map in shape [H, W].
        K (torch.Tensor): Camera intrinsic matrix in shape [3, 3].
        ext (torch.Tensor, optional): Camera extrinsic matrix in shape [4, 4]. Defaults to None, which will use the identity matrix.
        mask (torch.Tensor, optional): Foreground mask for extracting 3D points in shape [H, W]. Defaults to None.

    Returns:
        torch.Tensor: Extracted 3D points in shape [K, 3], where K is the number of unmasked points. If no mask is given, K=H*W.
    """
    H, W = depth.shape
    if ext is None:
        ext = torch.eye(4).to(depth)  # Use identity matrix if extrinsics are not provided

    # Create a grid of pixel coordinates in homogeneous form
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x = x.to(depth)
    y = y.to(depth)
    ones = torch.ones_like(x).to(depth)
    pixel_coords = torch.stack((x, y, ones), dim=-1).reshape(-1, 3).to(K)  # shape [H*W, 3]

    # Apply the intrinsics to get normalized camera coordinates
    K_inv = torch.inverse(K)
    normalized_coords = torch.mm(pixel_coords, K_inv.T) * depth.reshape(-1, 1)

    # Convert to 3D points in the camera frame
    camera_coords = torch.cat((normalized_coords, ones.reshape(-1, 1)), dim=-1)  # shape [H*W, 4]

    # Apply the extrinsic matrix to get coordinates in the world frame
    world_coords = torch.mm(camera_coords, ext.T.to(depth))[:, :3]  # shape [H*W, 3] ignoring the homogeneous coordinate

    # Apply mask if provided
    if mask is not None:
        masked_indices = torch.where(mask.reshape(-1) > 0)  # Get index of pixels where mask is non-zero
        pts = world_coords[masked_indices]
    else:
        pts = world_coords

    return pts

def get_interpolate_depth(depth_map, u, v):
    '''
    Interpolates the depth at specified non-integer pixel coordinates (u, v) using bilinear interpolation.

    Input:
        depth_map: torch.Tensor in shape [H, W]
        u, v: pixel coordinates in shape [N], torch.Tensor in floating points
    Return:
        corresponding depth value in pixel location with interpolation
    '''
    H, W = depth_map.shape
    max_h, max_w = H - 1, W - 1

    # Corners of the integer bounding box
    u0 = torch.floor(u).clamp(0, max_w)
    v0 = torch.floor(v).clamp(0, max_h)
    u1 = (u0 + 1).clamp(0, max_w)
    v1 = (v0 + 1).clamp(0, max_h)

    # Fractional parts
    u_frac = u - u0
    v_frac = v - v0

    # Gather the four nearest neighbors
    top_left = depth_map[v0.long(), u0.long()]
    top_right = depth_map[v0.long(), u1.long()]
    bottom_left = depth_map[v1.long(), u0.long()]
    bottom_right = depth_map[v1.long(), u1.long()]

    # Bilinear interpolation
    top = (1 - u_frac) * top_left + u_frac * top_right
    bottom = (1 - u_frac) * bottom_left + u_frac * bottom_right
    interpolated_depth = (1 - v_frac) * top + v_frac * bottom

    return interpolated_depth

def get_point_cloud_given_uv(depth, u, v, K):
    '''
    Retrieve the point cloud given depth map and the query pixel location u, v.

    Args:
        depth (torch.Tensor): Depth map in shape [H, W].
        u (torch.Tensor): Pixel coordinates along the width.
        v (torch.Tensor): Pixel coordinates along the height.
        K (torch.Tensor): Camera intrinsic matrix in shape [3, 3].

    Returns:
        pts (torch.Tensor): 3D points in the pixel location u, v.
    '''

    # Validate inputs
    assert u.size(0) == v.size(0), "u and v should have the same number of elements"
    assert K.size() == (3, 3), "Camera intrinsic matrix K must be 3x3"

    # Extract the intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Ensure the indices are within the bounds of the depth map
    H, W = depth.shape
    assert u.min() >= 0 and u.max() < W, "u-coordinate out of bounds"
    assert v.min() >= 0 and v.max() < H, "v-coordinate out of bounds"

    # Extract depth values at the specified u, v coordinates
    # selected_depths = depth[v, u]
    selected_depths = get_interpolate_depth(depth, u, v)

    # Convert image coordinates (u, v) into normalized camera coordinates
    x = (u.float() - cx) / fx * selected_depths
    y = (v.float() - cy) / fy * selected_depths
    z = selected_depths

    # Stack into a [N, 3] tensor where N is number of points
    pts = torch.stack((x, y, z), dim=1)

    return pts

def scale_alignment(depth_map1: torch.Tensor, depth_map2: torch.Tensor, pos1: torch.Tensor, pos2: torch.Tensor) -> float:
    '''
    Align the scale given two depth maps and corresponding matched positions.

    Args:
        depth_map1 (torch.Tensor): Depth map of the first frame.
        depth_map2 (torch.Tensor): Depth map of the second frame.
        pos1 (torch.Tensor): Pixel positions of matched points in the first frame in shape [N, 2].
        pos2 (torch.Tensor): Pixel positions of matched points in the second frame in shape [N, 2].

    Returns:
        scale (float): Scale factor between two frames obtained as median of per-point depth ratios.
                       The scale would typically be used as depth_map1 * scale_factor = depth_map2.
    '''

    # Ensure that pos1 and pos2 have integer coordinates suitable for indexing depth maps
    pos1 = pos1.long()
    pos2 = pos2.long()

    # Retrieve depth values at matched positions
    depth_values1 = depth_map1[pos1[:, 1], pos1[:, 0]]  # Indexing with y (row), x (column)
    depth_values2 = depth_map2[pos2[:, 1], pos2[:, 0]]

    # Handle zero depth values to prevent division by zero
    valid_mask = (depth_values1 > 0) & (depth_values2 > 0)
    valid_depths1 = depth_values1[valid_mask]
    valid_depths2 = depth_values2[valid_mask]

    # Calculate per-point scale factors as the ratio of depths
    scale_factors = valid_depths1 / valid_depths2

    # Compute the median scale factor to avoid influence of outliers
    if len(scale_factors) > 0:
        scale_factor = torch.median(scale_factors).item()
    else:
        scale_factor = 1.0  # Default to no scaling if no valid depths

    return scale_factor


def estimate_pose_ransac(pts_a, pts_b, K):
    """
    Estimate the pose between two sets of points from two images using the RANSAC algorithm.

    Parameters:
    pts_a (np.array): Coordinates of matched points in the first image, shape (N, 2)
    pts_b (np.array): Coordinates of matched points in the second image, shape (N, 2)
    K (np.array): The 3x3 intrinsic camera matrix

    Returns:
    R (np.array): The 3x3 rotation matrix
    t (np.array): The 3x1 translation vector
    mask (np.array): The mask of inliers computed by RANSAC (1=inlier, 0=outlier)
    """

    # Normalize the points using the intrinsic matrix
    pts_a_norm = cv2.undistortPoints(np.expand_dims(pts_a, axis=1), cameraMatrix=K, distCoeffs=None)
    pts_b_norm = cv2.undistortPoints(np.expand_dims(pts_b, axis=1), cameraMatrix=K, distCoeffs=None)

    # Estimate the Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts_a_norm, pts_b_norm, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1)

    # Decompose the Essential Matrix into rotation and translation
    _, R, t, _ = cv2.recoverPose(E, pts_a_norm, pts_b_norm, focal=1.0, pp=(0, 0))

    return R, t, mask

# def compute_dynamic_position(static_pos, motion_params):
#     dy_q = motion_params['quaternion']
#     dy_T = motion_params['translation'].unsqueeze(1)
#     pts_num = static_pos.shape[0]
#     homo_pos = torch.concat(
#         (static_pos, torch.ones([pts_num, 1], device=static_pos.device)),
#                             dim=1)
    
#     # construct transformation matrix
#     dy_rot = quat_to_rotmat(dy_q)
#     trans = torch.concat([dy_rot, dy_T], dim=1)
    
#     trans_homo = torch.zeros([dy_rot.shape[0], 4, 4]).to(dy_rot)
#     trans_homo[:, :, :3] = trans
#     trans_homo[:, -1, -1] = 1
    
    
#     pass