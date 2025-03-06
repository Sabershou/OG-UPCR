from typing import Optional
from .SC2_PCR import Matcher
import open3d as o3d
# from ..utils.transformations import transform_points_Rt
from .model_util import nn_gather
import numpy as np
import torch
from pytorch3d import transforms as pt3d_T
import time
import igraph
from scipy.spatial import cKDTree
import torch.nn.functional as F


@torch.jit.script
def invert_quaternion(quat):
    return quat * torch.tensor([1, -1, -1, -1]).to(quat)


def get_nearest_neighbor(
        q_points: np.ndarray,
        s_points: np.ndarray,
        return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1)
    if return_index:
        return distances, indices
    else:
        return distances


def get_nearest_neighbor_cuda(
        q_points,
        s_points,
):
    r"""Compute the nearest neighbor for the query points in support points."""

    distances = torch.cdist(q_points, s_points)
    nn_distances, _ = distances.min(dim=1)

    return nn_distances


def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.075):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


def compute_overlap_cuda(ref_points, src_points, transform=None, positive_radius=0.0375):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform_cuda(src_points, transform)
    nn_distances = get_nearest_neighbor_cuda(ref_points, src_points)
    overlap = (nn_distances < positive_radius).float().mean()
    return overlap


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def apply_transform_cuda(points, transform, normals=None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = torch.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = torch.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.int64,
    )
    return corr_indices


@torch.jit.script
def normalize_quaternion(quat):
    # deal with all 0s
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    w = quat[..., 0:1]
    w = torch.where(norm < 1e-9, w + 1, w)
    quat = torch.cat((w, quat[..., 1:]), dim=-1)

    # normalize
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    return quat


@torch.jit.script
def quaternion_distance(q0, q1):
    w_rel = (q0 * q1).sum(dim=1)
    w_rel = w_rel.clamp(min=-1, max=1)
    q_rel_error = 2 * w_rel.abs().acos()
    return q_rel_error


@torch.jit.script
def normalize_qt(params):
    t = params[:, 4:7]
    q = params[:, 0:4]
    q = normalize_quaternion(q)
    return torch.cat((q, t), dim=-1)


@torch.jit.script
def apply_quaternion(quaternion, point):
    # Copied over as is from pytorch3d; replaced wiht my invert
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = pt3d_T.quaternion_raw_multiply(
        pt3d_T.quaternion_raw_multiply(quaternion, point_as_quaternion),
        invert_quaternion(quaternion),
    )
    return out[..., 1:]


def random_qt(batch_size: int, q_mag: float, t_mag: float):
    assert q_mag >= 0.0 and q_mag < 1.57, "Rotation angle has to be between 0 and pi/2"

    # Random quaternion of magnitude theta (cos(theta/2
    h_mag = torch.ones(batch_size, 1) * q_mag / 2.0
    q_w = h_mag.cos()
    q_xyz = torch.randn(batch_size, 3)
    q_xyz = (q_xyz / q_xyz.norm(p=2, dim=1, keepdim=True)) * h_mag.sin()

    # get translation
    t = torch.randn(batch_size, 3)
    t = t / t.norm(p=2, dim=1, keepdim=True)

    param = torch.cat((q_w, q_xyz, t), dim=1)
    return param


@torch.jit.script
def transform_points_qt(
        points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, D = viewpoint.shape
    assert D == 7, "3 translation and 4 quat "

    q = viewpoint[:, None, 0:4]
    t = viewpoint[:, None, 4:7]

    # Normalize quaternion
    q = normalize_quaternion(q)

    if inverse:
        # translate then rotate
        points = points - t
        points = apply_quaternion(invert_quaternion(q), points)
    else:
        # rotate then invert
        points = apply_quaternion(q, points)
        points = points + t

    return points


@torch.jit.script
def transform_points_Rt(
        points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, H, W = viewpoint.shape
    assert H == 3 and W == 4, "Rt is B x 3 x 4 "
    t = viewpoint[:, :, 3]
    r = viewpoint[:, :, 0:3]

    # transpose r to handle the fact that P in num_points x 3
    # yT = (RX)T = XT @ RT
    r = r.transpose(1, 2).contiguous()

    # invert if needed
    if inverse:
        points = points - t[:, None, :]
        points = points.bmm(r.inverse())
    else:
        points = points.bmm(r)
        points = points + t[:, None, :]

    return points


if __name__ == "__main__":
    rand_pt = torch.randn(4, 1000, 3)

    rand_qt = random_qt(4, 0.5, 3)

    # qt -> rt
    q = rand_qt[:, :4]
    t = rand_qt[:, 4:, None]

    R = pt3d_T.quaternion_to_matrix(q)
    Rinv = pt3d_T.quaternion_to_matrix(pt3d_T.quaternion_invert(q))

    Rti = torch.cat((Rinv, t), dim=2)
    Rt = torch.cat((R, t), dim=2)

    rot_qt = transform_points_qt(rand_pt, rand_qt)
    rot_Rt = transform_points_Rt(rand_pt, Rt)
    rot_Rti = transform_points_Rt(rand_pt, Rti)

    qt_Rt = (rot_qt - rot_Rt).norm(dim=2, p=2).mean()
    qt_Rti = (rot_qt - rot_Rti).norm(dim=2, p=2).mean()
    Rt_Rti = (rot_Rti - rot_Rt).norm(dim=2, p=2).mean()

    print(f"|| points ||:    {rand_pt.norm(p=2, dim=2).mean()}")
    print(f"Diff Rt and qt:  {qt_Rt:.4e}")
    print(f"Diff Rti and qt: {qt_Rti:.4e}")
    print(f"Diff Rti and Rt: {Rt_Rti:.4e}")


def align(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)

    Rt = randomized_weighted_procrustes(corr_P, corr_Q, weights, align_cfg)

    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(corr_P, Rt)
    dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)

    weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    corr_loss = (weights_norm * dist_PQ).sum(dim=1)

    return Rt, corr_loss


def align_ransac(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)

    Rt = registration_with_ransac_from_correspondences(corr_P, corr_Q, distance_threshold=0.05, ransac_n=3,
                                                       num_iterations=100000)
    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(corr_P, Rt)
    dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)

    weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    corr_loss = (weights_norm * dist_PQ).sum(dim=1)

    return Rt, corr_loss


def align_mac(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)[:, :180, :]
    corr_Q = nn_gather(Q, corr_Q_idx)[:, :180, :]
    weights = weights[:, :180]

    Rt = registration_with_mac_from_correspondences(corr_P.squeeze(0), corr_Q.squeeze(0))
    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(corr_P, Rt)
    dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)

    weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    corr_loss = (weights_norm * dist_PQ).sum(dim=1)

    return Rt, corr_loss


def align_SC2PCR(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres
    # corr_idx = torch.cat((corr_P_idx, corr_Q_idx), dim=0)
    N = 2
    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)

    Rt = randomized_weighted_procrustes(corr_P, corr_Q, weights, align_cfg)
    # overlap = 0
    src_points = P.detach().squeeze(0).cpu().numpy()
    ref_points = Q.detach().squeeze(0).cpu().numpy()
    corr_P_idx_new = corr_P_idx.detach()
    corr_Q_idx_new = corr_Q_idx.detach()
    weights_new = weights.detach()
    overlap_ratio = 0.5
    for i in range(N):
        overlap_ratio = compute_overlap(ref_points, src_points, transform=Rt.detach().squeeze(0).cpu().numpy())
        if overlap_ratio < 0.1:
            corr_indices = get_correspondences(ref_points, src_points,
                                               Rt.detach().squeeze(0).cpu().numpy(), 0.0375)
            if len(corr_indices) == 0:
                break
            est_ref_idx = torch.from_numpy(np.array(list(set(corr_indices[:, 0].tolist())))).cuda()
            est_src_idx = torch.from_numpy(np.array(list(set(corr_indices[:, 1].tolist())))).cuda()
            src_mask = torch.isin(corr_P_idx_new[0], est_src_idx).unsqueeze(0)
            ref_mask = torch.isin(corr_Q_idx_new[0], est_ref_idx).unsqueeze(0)
            mask = src_mask.logical_and(ref_mask)
            if torch.sum(~mask) < 100:
                break
            corr_P_idx_new = corr_P_idx_new[~mask].unsqueeze(0)
            corr_Q_idx_new = corr_Q_idx_new[~mask].unsqueeze(0)
            weights_new = weights_new[~mask].unsqueeze(0)
            corr_P_new = nn_gather(P, corr_P_idx_new)
            corr_Q_new = nn_gather(Q, corr_Q_idx_new)
            Rt = randomized_weighted_procrustes(corr_P_new, corr_Q_new, weights_new, align_cfg)
            # corr_P_idx_new = corr_P_idx[~torch.isin(corr_P_idx, est_src_idx)]
            # corr_Q_idx_new = corr_Q_idx[~torch.isin(corr_Q_idx, est_ref_idx)]
            # src_indices = torch.from_numpy(corr_indices[:, 0]).cuda()
            # ref_indices = torch.from_numpy(corr_indices[:, 1]).cuda()
        else:
            corr_indices = get_correspondences(ref_points, src_points,
                                               Rt.detach().squeeze(0).cpu().numpy(), 0.001)
            if len(corr_indices) == 0:
                break
            est_ref_idx = torch.from_numpy(np.array(list(set(corr_indices[:, 0].tolist())))).cuda()
            est_src_idx = torch.from_numpy(np.array(list(set(corr_indices[:, 1].tolist())))).cuda()
            # ref_indices = torch.from_numpy(corr_indices[:, 0]).cuda()
            # src_indices = torch.from_numpy(corr_indices[:, 1]).cuda()
            src_mask = torch.isin(corr_P_idx_new[0], est_src_idx).unsqueeze(0)
            ref_mask = torch.isin(corr_Q_idx_new[0], est_ref_idx).unsqueeze(0)
            mask = src_mask.logical_and(ref_mask)
            mask_number = torch.sum(mask)
            if mask_number < 100:
                break
            corr_P_idx_new = corr_P_idx_new[mask].unsqueeze(0)
            corr_Q_idx_new = corr_Q_idx_new[mask].unsqueeze(0)
            weights_new = weights_new[mask].unsqueeze(0)
            break

    corr_P = nn_gather(P, corr_P_idx_new)
    corr_Q = nn_gather(Q, corr_Q_idx_new)
    matcher = Matcher(inlier_threshold=0.1, num_node=5000, use_mutual=False, d_thre=0.1, num_iterations=10, ratio=0.95,
                      nms_radius=0.1, max_points=5000, k1=30, k2=20)
    Rt_new, seeds = matcher.SC2_PCR(corr_P, corr_Q, overlap_ratio)
    Rt_new = Rt_new[:, :3, :]

    # Calculate correspondance loss
    # corr_P_rot = transform_points_Rt(corr_P, Rt)
    corr_P_rot_new = transform_points_Rt(corr_P, Rt_new)
    # dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)
    dist_PQ_new = (corr_P_rot_new - corr_Q).norm(p=2, dim=2)
    weights_norm = weights_new / weights_new.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # corr_loss = (weights_norm * dist_PQ).sum(dim=1)
    corr_loss_new = (weights_norm * dist_PQ_new).sum(dim=1)
    # if corr_loss_new < corr_loss:
    #     Rt = Rt_new
    #     corr_loss = corr_loss_new
    return Rt_new, corr_loss_new, seeds


def align_IPID(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    num_points = 200
    corr_P_idx, corr_Q_idx, weights, _ = corres
    # corr_idx = torch.cat((corr_P_idx, corr_Q_idx), dim=0)
    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)
    src_points = P.detach()
    ref_points = Q.detach()
    corr_P_idx_new = corr_P_idx.detach()
    corr_Q_idx_new = corr_Q_idx.detach()
    weights_new = weights
    # weights_max, weights_mean, weights_med = torch.max(weights), torch.mean(weights), torch.median(weights)
    # weights_new = correspondence_weights_optimization(src_points, ref_points, corr_P, corr_Q, weights, align_cfg)
    weights_new, weights_idx = torch.topk(weights_new, k=num_points * 2, dim=1)

    # weights_max, weights_mean, weights_med = torch.max(weights_new), torch.mean(weights_new), torch.median(weights_new)
    # filtered_mask = weights_new > weights_mean
    # corr_P_idx_new = corr_P_idx_new[filtered_mask].unsqueeze(0)
    # corr_Q_idx_new = corr_Q_idx_new[filtered_mask].unsqueeze(0)
    # weights_new = weights_new[filtered_mask].unsqueeze(0)

    corr_P_idx_new = corr_P_idx[0, weights_idx]
    corr_Q_idx_new = corr_Q_idx[0, weights_idx]
    corr_P = nn_gather(P, corr_P_idx_new)
    corr_Q = nn_gather(Q, corr_Q_idx_new)
    # corr_P_new, corr_Q_new, weights_o, Rt_new = spatial_consistency_constraint(corr_P, corr_Q, weights_new, d_thr=0.1,
    #                                                                            k=0.2)
    # Rt_new = randomized_weighted_procrustes(corr_P, corr_Q, weights_new, align_cfg)
    Rt_new = registration_with_ransac_from_correspondences(corr_P, corr_Q, distance_threshold=0.10, ransac_n=3,
                                                           num_iterations=100000)
    # Rt_new = post_refinement(Rt_new, corr_P, corr_Q, 20)
    # SC2PCR
    # matcher = Matcher(inlier_threshold=0.1, num_node=5000, use_mutual=False, d_thre=0.1, num_iterations=10, ratio=0.5,
    #                   nms_radius=0.1, max_points=10000, k1=30, k2=20)
    # Rt_new, seeds = matcher.SC2_PCR(corr_P, corr_Q)
    Rt_new = Rt_new[:, :3, :]

    # Calculate correspondance loss
    # corr_P_rot = transform_points_Rt(corr_P, Rt)
    corr_P_rot_new = transform_points_Rt(corr_P, Rt_new)
    # dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)
    dist_PQ_new = (corr_P_rot_new - corr_Q).norm(p=2, dim=2)
    weights_norm = weights_new / weights_new.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # corr_loss = (weights_norm * dist_PQ).sum(dim=1)
    corr_loss_new = (weights_norm * dist_PQ_new).sum(dim=1)
    # if corr_loss_new < corr_loss:
    #     Rt = Rt_new
    #     corr_loss = corr_loss_new
    return Rt_new, corr_loss_new


def correspondence_weights_optimization(src_points, ref_points, pts_ref, pts_tar, weights, align_cfg):
    """
    Input:
        pts_ref     FloatTensor (N x C x 3)     reference points
        pts_tar     FloatTensor (N x C x 3)     target points
        weights     FloatTensor (N x C)         weights for each correspondance
        align_cfg   YACS config                 alignment configuration

    Returns:        FloatTensor (N x 3 x 4)     Esimated Transform ref -> tar
    """
    # Define/initialize some key variables and cfgs
    batch_size, num_pts, _ = pts_ref.shape

    # Do the SVD optimization on N subsets
    # N = align_cfg.num_seeds
    N = 10
    # subset = align_cfg.point_ratio
    subset = 0.20

    if subset < 1.0:
        num_matches = int(subset * num_pts)
        indices = torch.LongTensor(batch_size, N, num_matches).to(device=pts_ref.device)
    else:
        num_matches = num_pts

    # get a subset of points and detach
    pts_ref_c = pts_ref.unsqueeze(1).repeat(1, N, 1, 1)
    pts_tar_c = pts_tar.unsqueeze(1).repeat(1, N, 1, 1)
    if subset < 1.0:
        indices.random_(num_pts)
        pts_ref_c = pts_ref_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        pts_tar_c = pts_tar_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
            weights_c = weights_c.gather(2, indices)
        else:
            weights_c = None

    else:
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
        else:
            weights_c = None

    # reshape to batch x N --- basically a more manual (and inefficient) vmap, right?!

    pts_ref_c = pts_ref_c.view(batch_size * N, num_matches, 3).contiguous()
    pts_tar_c = pts_tar_c.view(batch_size * N, num_matches, 3).contiguous()
    weights_c = weights_c.view(batch_size * N, num_matches).contiguous()
    # weights_c = F.normalize(weights_c, p=2, dim=1)
    # weights_max, weights_min, weights_med = torch.max(weights_c), torch.min(weights_c), torch.median(weights_c)

    # Initialize VP
    Rt = paired_svd(pts_ref_c, pts_tar_c, weights_c)
    Rt = Rt.view(batch_size, N, 3, 4).contiguous()

    # Iterate over random subsets/seeds -- should be sped up somehow.
    for k in range(N):
        # calculate chamfer loss for back prop
        c_Rt = Rt[:, k]
        c_indices = indices[:, k]
        # overlap_ratio = compute_overlap(ref_points.detach().squeeze(0).cpu().numpy(),
        #                                 src_points.detach().squeeze(0).cpu().numpy(),
        #                                 transform=c_Rt.detach().squeeze(0).cpu().numpy())
        overlap_ratio = compute_overlap_cuda(ref_points.squeeze(0), src_points.squeeze(0), transform=c_Rt.squeeze(0))
        if overlap_ratio > 0.1:
            weights[0, c_indices] *= 2.0
            # weights_c[k] = weights_c[k] * 1.4
            # weights.scatter_(dim=1, index=c_indices, src=weights_c[k].unsqueeze(0))
        else:
            weights[0, c_indices] *= 0.5
            # weights_c[k] = weights_c[k] * 0.8
            # weights.scatter_(dim=1, index=c_indices, src=weights_c[k].unsqueeze(0))

    return weights


def randomized_weighted_procrustes(pts_ref, pts_tar, weights, align_cfg):
    """
    Adapts the Weighted Procrustes algorithm (Choy et al, CVPR 2020) to subsets.
    Specifically, the algorithm randomly samples N subsets and applies the weighted
    procrustes algorithm to it. It then picks the solution that minimzies the chamfer
    distances over all the correspondences.

    Input:
        pts_ref     FloatTensor (N x C x 3)     reference points
        pts_tar     FloatTensor (N x C x 3)     target points
        weights     FloatTensor (N x C)         weights for each correspondance
        align_cfg   YACS config                 alignment configuration

    Returns:        FloatTensor (N x 3 x 4)     Esimated Transform ref -> tar
    """
    # Define/initialize some key variables and cfgs
    batch_size, num_pts, _ = pts_ref.shape

    # Do the SVD optimization on N subsets
    # N = align_cfg.num_seeds
    N = 100
    subset = align_cfg.point_ratio

    if subset < 1.0:
        num_matches = int(subset * num_pts)
        indices = torch.LongTensor(batch_size, N, num_matches).to(device=pts_ref.device)
    else:
        num_matches = num_pts

    # get a subset of points and detach
    pts_ref_c = pts_ref.unsqueeze(1).repeat(1, N, 1, 1)
    pts_tar_c = pts_tar.unsqueeze(1).repeat(1, N, 1, 1)
    if subset < 1.0:
        indices.random_(num_pts)
        pts_ref_c = pts_ref_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        pts_tar_c = pts_tar_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
            weights_c = weights_c.gather(2, indices)
        else:
            weights_c = None

    else:
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
        else:
            weights_c = None

    # reshape to batch x N --- basically a more manual (and inefficient) vmap, right?!
    pts_ref_c = pts_ref_c.view(batch_size * N, num_matches, 3).contiguous()
    pts_tar_c = pts_tar_c.view(batch_size * N, num_matches, 3).contiguous()
    weights_c = weights_c.view(batch_size * N, num_matches).contiguous()

    # Initialize VP
    Rt = paired_svd(pts_ref_c, pts_tar_c, weights_c)
    Rt = Rt.view(batch_size, N, 3, 4).contiguous()

    best_loss = 1e10 * torch.ones(batch_size).to(pts_ref)
    best_seed = -1 * torch.ones(batch_size).int()

    # Iterate over random subsets/seeds -- should be sped up somehow.
    # We're finding how each estimate performs for all the correspondances and picking
    # the one that achieves the best weighted chamfer error
    for k in range(N):
        # calculate chamfer loss for back prop
        c_Rt = Rt[:, k]
        pts_ref_rot = transform_points_Rt(pts_ref, c_Rt, inverse=False)
        c_chamfer = (pts_ref_rot - pts_tar).norm(dim=2, p=2)

        if weights is not None:
            c_chamfer = weights * c_chamfer

        c_chamfer = c_chamfer.mean(dim=1)

        # Find the better indices, and update best_loss and best_seed
        better_indices = (c_chamfer < best_loss).detach()
        best_loss[better_indices] = c_chamfer[better_indices]
        best_seed[better_indices] = k

    # convert qt to Rt
    Rt = Rt[torch.arange(batch_size), best_seed.long()]
    return Rt


def make_open3d_point_cloud(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def registration_with_ransac_from_correspondences(
        src_points,
        ref_points,
        correspondences=None,
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=100000,
        GTmat=None,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pts = src_points.squeeze(0).cpu().numpy()
    ref_pts = ref_points.squeeze(0).cpu().numpy()
    src_pcd = make_open3d_point_cloud(src_pts)
    ref_pcd = make_open3d_point_cloud(ref_pts)

    if correspondences is None:
        indices = np.arange(src_pts.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, num_iterations),
    )

    # final_trans = torch.from_numpy(result.transformation).float()
    # re, te = transformation_error(final_trans, GTmat)
    # final_trans1 = post_refinement(initial_trans=final_trans[None], src_kpts=src_pts[None],
    #                                tgt_kpts=tgt_pts[None],
    #                                iters=30)
    # re1, te1 = transformation_error(final_trans1[0], GTmat)
    # if re1 <= re and te1 <= te:
    #     final_trans = final_trans1[0]
    #     re, te = re1, te1
    # final_trans = final_trans.detach().cpu().numpy()

    return torch.from_numpy(result.transformation).unsqueeze(0)[:, :3, :].cuda().to(torch.float32)


@torch.jit.script
def paired_svd(X, Y, weights: Optional[torch.Tensor] = None):
    """
    The core part of the (Weighted) Procrustes algorithm. Esimate the transformation
    using an SVD.

    Input:
        X           FloatTensor (B x N x 3)     XYZ for source point cloud
        Y           FloatTensor (B x N x 3)     XYZ for target point cloud
        weights     FloatTensor (B x N)         weights for each correspondeance

    return          FloatTensor (B x 3 x 4)     Rt transformation
    """

    # It's been advised to turn into double to avoid numerical instability with SVD
    X = X.double()
    Y = Y.double()

    if weights is not None:
        eps = 1e-5
        weights = weights.double()
        weights = weights.unsqueeze(2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        X_mean = (X * weights).sum(dim=1, keepdim=True)
        Y_mean = (Y * weights).sum(dim=1, keepdim=True)
        X_c = weights * (X - X_mean)
        Y_c = weights * (Y - Y_mean)
    else:
        X_mean = X.mean(dim=1, keepdim=True)
        Y_mean = Y.mean(dim=1, keepdim=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean

    # Reflection to handle numerically instable COV matrices
    reflect = torch.eye(3).to(X)
    reflect[2, 2] = -1

    # Calculate H Matrix.
    H = torch.matmul(X_c.transpose(1, 2).contiguous(), Y_c)

    # Compute SVD
    U, S, V = torch.svd(H)

    # Compute R
    U_t = U.transpose(2, 1).contiguous()
    R = torch.matmul(V, U_t)

    # Reflect R for determinant less than 0
    R_det = torch.det(R)
    V_ref = torch.matmul(V, reflect[None, :, :])
    R_ref = torch.matmul(V_ref, U_t)
    R = torch.where(R_det[:, None, None] < 0, R_ref, R)

    # Calculate t
    t = Y_mean[:, 0, :, None] - torch.matmul(R, X_mean[:, 0, :, None])
    Rt = torch.cat((R, t[:, :, 0:1]), dim=2)
    return Rt.float()


def registration_with_mac_from_correspondences(src_pts, tgt_pts):
    # src_pts = torch.from_numpy(src_pts)
    # tgt_pts = torch.from_numpy(tgt_pts)
    corr_data = torch.cat((src_pts, tgt_pts), dim=1)
    t1 = time.perf_counter()
    src_dist = ((src_pts[:, None, :] - src_pts[None, :, :]) ** 2).sum(-1) ** 0.5
    tgt_dist = ((tgt_pts[:, None, :] - tgt_pts[None, :, :]) ** 2).sum(-1) ** 0.5
    cross_dis = torch.abs(src_dist - tgt_dist)
    FCG = torch.clamp(1 - cross_dis ** 2 / 0.1 ** 2, min=0)
    FCG = FCG - torch.diag_embed(torch.diag(FCG))
    FCG[FCG < 0.99] = 0
    SCG = torch.matmul(FCG, FCG) * FCG
    t2 = time.perf_counter()
    # print(f'Graph construction: %.2fms' % ((t2 - t1) * 1000))

    SCG = SCG.cpu().numpy()
    t1 = time.perf_counter()
    graph = igraph.Graph.Adjacency((SCG > 0).tolist())
    graph.es['weight'] = SCG[SCG.nonzero()]
    graph.vs['label'] = range(0, corr_data.shape[0])
    graph.to_undirected()
    macs = graph.maximal_cliques(min=3)
    t2 = time.perf_counter()
    # print(f'Search maximal cliques: %.2fms' % ((t2 - t1) * 1000))
    # print(f'Total: %d' % len(macs))
    clique_weight = np.zeros(len(macs), dtype=float)
    for ind in range(len(macs)):
        mac = list(macs[ind])
        if len(mac) >= 3:
            for i in range(len(mac)):
                for j in range(i + 1, len(mac)):
                    clique_weight[ind] = clique_weight[ind] + SCG[mac[i], mac[j]]

    clique_ind_of_node = np.ones(corr_data.shape[0], dtype=int) * -1
    search_value = -1
    max_clique_weight = np.zeros(corr_data.shape[0], dtype=float)
    max_size = 3
    for ind in range(len(macs)):
        mac = list(macs[ind])
        weight = clique_weight[ind]
        if weight > 0:
            for i in range(len(mac)):
                if weight > max_clique_weight[mac[i]]:
                    max_clique_weight[mac[i]] = weight
                    clique_ind_of_node[mac[i]] = ind
                    max_size = len(mac) > max_size and len(mac) or max_size

    filtered_clique_ind = list(set(clique_ind_of_node))
    if search_value in filtered_clique_ind:
        filtered_clique_ind.remove(-1)
    # print(f'After filtered: %d' % len(filtered_clique_ind))

    group = []
    for s in range(3, max_size + 1):
        group.append([])
    for ind in filtered_clique_ind:
        mac = list(macs[ind])
        group[len(mac) - 3].append(ind)

    tensor_list_A = []
    tensor_list_B = []
    for i in range(len(group)):
        if len(group[i]) == 0:
            continue
        batch_A = src_pts[list(macs[group[i][0]])][None]
        batch_B = tgt_pts[list(macs[group[i][0]])][None]
        if len(group) == 1:
            continue
        for j in range(1, len(group[i])):
            mac = list(macs[group[i][j]])
            src_corr = src_pts[mac][None]
            tgt_corr = tgt_pts[mac][None]
            batch_A = torch.cat((batch_A, src_corr), 0)
            batch_B = torch.cat((batch_B, tgt_corr), 0)
        tensor_list_A.append(batch_A)
        tensor_list_B.append(batch_B)

    inlier_threshold = 0.1
    max_score = 0
    final_trans = torch.eye(4).cuda()
    for i in range(len(tensor_list_A)):
        trans = rigid_transform_3d(tensor_list_A[i], tensor_list_B[i], None, 0)
        pred_tgt = transform(src_pts[None], trans)  # [bs, num_corr, 3]
        L2_dis = torch.norm(pred_tgt - tgt_pts[None], dim=-1)  # [bs, num_corr]
        MAE_score = torch.div(torch.sub(inlier_threshold, L2_dis), inlier_threshold)
        MAE_score = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)
        max_batch_score_ind = MAE_score.argmax(dim=-1)
        max_batch_score = MAE_score[max_batch_score_ind]
        if max_batch_score > max_score:
            max_score = max_batch_score
            final_trans = trans[max_batch_score_ind]

    # re, te = transformation_error(final_trans, GTmat)
    final_trans1 = post_refinement(initial_trans=final_trans[None], src_kpts=src_pts[None], tgt_kpts=tgt_pts[None],
                                   iters=20)
    # re1, te1 = transformation_error(final_trans1[0], GTmat)
    # if re1 <= re and te1 <= te:
    #     final_trans = final_trans1[0]
    #     re, te = re1, te1

    return final_trans1[:, :3, :]


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)  # 升维度，然后变为对角阵
    H = Am.permute(0, 2, 1) @ Weight @ Bm  # permute : tensor中的每一块做转置

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)


def transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3],
                                 pts.permute(0, 2, 1)) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum('nm,mk->nk', trans[:3, :3],
                                 pts.T) + trans[:3, 3:4]
        return trans_pts.T


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def mac_post_refinement(initial_trans, src_kpts, tgt_kpts, iters, weights=None):
    inlier_threshold = 0.1
    pre_inlier_count = 0
    for i in range(iters):
        pred_tgt = transform(src_kpts, initial_trans)
        L2_dis = torch.norm(pred_tgt - tgt_kpts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]
        inlier_count = torch.sum(pred_inlier)
        if inlier_count <= pre_inlier_count:
            break
        pre_inlier_count = inlier_count
        initial_trans = rigid_transform_3d(
            A=src_kpts[:, pred_inlier, :],
            B=tgt_kpts[:, pred_inlier, :],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier]
        )
    return initial_trans


def spatial_consistency_constraint(src_keypts, tgt_keypts, weights, d_thr, k):
    bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]
    max_points = 10000
    #################################
    # downsample points
    #################################
    if num_corr > max_points:
        src_keypts = src_keypts[:, :max_points, :]
        tgt_keypts = tgt_keypts[:, :max_points, :]
        num_corr = max_points

    #################################
    # compute cross dist
    #################################
    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)

    #################################
    # compute first order measure
    #################################
    SC_dist_thre = d_thr
    SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
    hard_SC_measure = (cross_dist < SC_dist_thre).float()
    # SC_measure[:, torch.arange(SC_measure.shape[1]), torch.arange(SC_measure.shape[1])] = 0

    #################################
    # select reliable seed correspondences
    #################################
    # confidence = cal_leading_eigenvector(SC_measure, method='power')
    confidence = weights
    nms_radius = 0.1
    # k = 0.1
    seeds = pick_seeds(src_dist, confidence, R=nms_radius, max_num=int(num_corr * k))
    SC2_dist_thre = d_thr / 2
    hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
    seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                                  index=seeds[:, :, None].expand(-1, -1, num_corr))
    seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                                              index=seeds[:, :, None].expand(-1, -1, num_corr))
    SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure

    src_knn, tgt_knn, total_weight, final_trans = cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts, k1=30,
                                                                 k2=10, d_thr=d_thr, inlier_threshold=0.1)
    #################################
    # refine the result by recomputing the transformation over the whole set
    #################################
    final_trans = post_refinement(final_trans, src_keypts, tgt_keypts, 20)
    src_kpts, tgt_kpts = src_knn.view([bs, -1, 3]), tgt_knn.view([bs, -1, 3])
    total_weight = total_weight.view([bs, -1])

    return src_kpts, tgt_kpts, total_weight, final_trans


def cal_leading_eigenvector(M, method='power'):
    """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
    """

    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(10):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)


def pick_seeds(dists, scores, R, max_num):
    """
    Select seeding points using Non Maximum Suppression. (here we only support bs=1)
    Input:
        - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
        - scores:      [bs, num_corr]     initial confidence of each correspondence
        - R:           float              radius of nms
        - max_num:     int                maximum number of returned seeds
    Output:
        - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
    """
    assert scores.shape[0] == 1

    # parallel Non Maximum Suppression (more efficient)
    score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
    # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
    score_relation = score_relation.bool() | (dists[0] >= R).bool()
    is_local_max = score_relation.min(-1)[0].float()

    score_local_max = scores * is_local_max
    sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

    # max_num = scores.shape[1]

    return_idx = sorted_score[:, 0: max_num].detach()

    return return_idx


def cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts, k1, k2, d_thr, inlier_threshold):
    """
    Calculate the transformation for each seeding correspondences.
    Input:
        - seeds:         [bs, num_seeds]              the index to the seeding correspondence
        - SC2_measure: [bs, num_corr, num_channels]
        - src_keypts:    [bs, num_corr, 3]
        - tgt_keypts:    [bs, num_corr, 3]
    Output: leading eigenvector
        - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
    """
    bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]

    if k1 > num_channels:
        k1 = 4
        k2 = 4

    #################################
    # The first stage consensus set sampling
    # Finding the k1 nearest neighbors around each seed
    #################################
    sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
    knn_idx = sorted_score[:, :, 0: k1]
    sorted_value, _ = torch.sort(SC2_measure, dim=2, descending=True)
    idx_tmp = knn_idx.contiguous().view([bs, -1])
    idx_tmp = idx_tmp[:, :, None]
    idx_tmp = idx_tmp.expand(-1, -1, 3)

    #################################
    # construct the local SC2 measure of each consensus subset obtained in the first stage.
    #################################
    src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])  # [bs, num_seeds, k, 3]
    tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
    src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
    tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
    cross_dist = torch.abs(src_dist - tgt_dist)
    local_hard_SC_measure = (cross_dist < d_thr).float()
    local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

    #################################
    # perform second stage consensus set sampling
    #################################
    sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
    knn_idx_fine = sorted_score[:, :, :, 0: k2]

    #################################
    # construct the soft SC2 matrix of the consensus set
    #################################
    num = knn_idx_fine.shape[1]
    knn_idx_fine = knn_idx_fine.contiguous().view([bs, num, -1])[:, :, :, None]
    knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
    src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])  # [bs, num_seeds, k, 3]
    tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])

    src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
    tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
    cross_dist = torch.abs(src_dist - tgt_dist)
    # local_hard_measure = (cross_dist < self.d_thre * 2).float()
    # local_SC2_measure = torch.matmul(local_hard_measure, local_hard_measure) / k2
    local_SC_measure = torch.clamp(1 - cross_dist ** 2 / d_thr ** 2, min=0)
    # local_SC2_measure = local_SC_measure * local_SC2_measure
    local_SC2_measure = local_SC_measure
    local_SC2_measure = local_SC2_measure.view([-1, k2, k2])

    #################################
    # Power iteratation to get the inlier probability
    #################################
    local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
    total_weight = cal_leading_eigenvector(local_SC2_measure, method='power')
    total_weight = total_weight.view([bs, -1, k2])
    total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

    #################################
    # calculate the transformation by weighted least-squares for each subsets in parallel
    #################################
    total_weight = total_weight.view([-1, k2])
    src_knn = src_knn_fine
    tgt_knn = tgt_knn_fine
    src_knn, tgt_knn = src_knn.view([-1, k2, 3]), tgt_knn.view([-1, k2, 3])

    #################################
    # compute the rigid transformation for each seed by the weighted SVD
    #################################
    seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
    seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

    #################################
    # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
    #################################
    pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                 src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                3:4]  # [bs, num_seeds, num_corr, 3]
    #################################
    # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
    #################################
    pred_position = pred_position.permute(0, 1, 3, 2)
    L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
    seedwise_fitness = torch.sum((L2_dis < inlier_threshold).float(), dim=-1)  # [bs, num_seeds]
    batch_best_guess = seedwise_fitness.argmax(dim=1)
    best_guess_ratio = seedwise_fitness[0, batch_best_guess]
    final_trans = seedwise_trans.gather(dim=1,
                                        index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

    return src_knn, tgt_knn, total_weight, final_trans


def post_refinement(initial_trans, src_keypts, tgt_keypts, it_num, weights=None):
    """
    Perform post refinement using the initial transformation matrix, only adopted during testing.
    Input
        - initial_trans: [bs, 4, 4]
        - src_keypts:    [bs, num_corr, 3]
        - tgt_keypts:    [bs, num_corr, 3]
        - weights:       [bs, num_corr]
    Output:
        - final_trans:   [bs, 4, 4]
    """
    assert initial_trans.shape[0] == 1
    inlier_threshold = 0.10

    # inlier_threshold_list = [self.inlier_threshold] * it_num

    if inlier_threshold == 0.10:  # for 3DMatch
        inlier_threshold_list = [0.10] * it_num
    else:  # for KITTI
        inlier_threshold_list = [1.2] * it_num

    previous_inlier_num = 0
    for inlier_threshold in inlier_threshold_list:
        warped_src_keypts = transform(src_keypts, initial_trans)

        L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
        inlier_num = torch.sum(pred_inlier)
        if abs(int(inlier_num - previous_inlier_num)) < 1:
            break
        else:
            previous_inlier_num = inlier_num
        initial_trans = rigid_transform_3d(
            A=src_keypts[:, pred_inlier, :],
            B=tgt_keypts[:, pred_inlier, :],
            # weights=None,
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
            # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
        )
    return initial_trans
