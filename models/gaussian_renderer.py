from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import torch
import torch.nn.functional as F
from .recon_helpers import setup_camera
import numpy as np
from .model_util import build_rotation, rotation_matrix_to_quaternion
from scipy.spatial.transform import Rotation


def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette


def initialize_first_timestep(color, depth, intrinsics, pose, pose_first, num_frames, scene_radius_depth_ratio=3,
                              mean_sq_dist_method='projective', densify_dataset=None):
    # Process RGB-D Data
    # 将颜色数据调整为PyTorch的形状和范围。
    # color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    # # 调整深度数据的形状。
    # depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Process Camera Parameters
    # 提取相机内参并计算相机到世界坐标系的逆矩阵。
    # B = color.shape[0]
    intrinsics = intrinsics[:3, :3]
    w2c = pose
    first_w2c = pose_first

    # Setup Camera
    # 使用提取的相机参数设置相机。
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_w2c.detach().cpu().numpy())

    if densify_dataset is not None:  # 如果提供了密集化数据集，获取第一帧RGB-D数据和相机内参，并进行相应的处理。
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   first_w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0)  # Mask out invalid depth values
    mask = mask.reshape(-1)
    # 根据颜色、深度、相机内参、相机到世界坐标系的逆矩阵等信息，使用 get_pointcloud 函数获取初始点云。
    # 通过 mask 过滤掉无效深度值。
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)  # init_pt_cld是对每个点都生成的点云吗

    # Initialize Parameters
    # 利用初始点云和其他信息，使用 initialize_params 函数初始化模型参数和变量
    # params里是生成点云的点位置颜色还有mean3等,variable则是又点云中的点生成的3d高斯性质,如2d半径等
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    # 估计场景半径，用于后续的高斯光斑密集化。
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, first_w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, first_w2c, cam


def initialize_new_timestep(color, depth, intrinsics, pose, pose_new, num_frames, scene_radius_depth_ratio=3,
                            mean_sq_dist_method='projective', densify_dataset=None):
    # Process RGB-D Data
    # 将颜色数据调整为PyTorch的形状和范围。
    # color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    # # 调整深度数据的形状。
    # depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Process Camera Parameters
    # 提取相机内参并计算相机到世界坐标系的逆矩阵。
    # B = color.shape[0]
    intrinsics = intrinsics[:3, :3]
    w2c = pose
    new_w2c = pose_new

    # Setup Camera
    # 使用提取的相机参数设置相机。
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), new_w2c.detach().cpu().numpy())

    if densify_dataset is not None:  # 如果提供了密集化数据集，获取第一帧RGB-D数据和相机内参，并进行相应的处理。
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   new_w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0)  # Mask out invalid depth values
    mask = mask.reshape(-1)
    # 根据颜色、深度、相机内参、相机到世界坐标系的逆矩阵等信息，使用 get_pointcloud 函数获取初始点云。
    # 通过 mask 过滤掉无效深度值。
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)  # init_pt_cld是对每个点都生成的点云吗

    # Initialize Parameters
    # 利用初始点云和其他信息，使用 initialize_params 函数初始化模型参数和变量
    # params里是生成点云的点位置颜色还有mean3等,variable则是又点云中的点生成的3d高斯性质,如2d半径等
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    # 估计场景半径，用于后续的高斯光斑密集化。
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, new_w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, new_w2c, cam


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    # 从颜色图像中提取宽度和高度，并计算相机内参的各个分量。
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    # 计算像素坐标和深度信息：

    # 利用网格生成像素坐标 xx 和 yy。
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    # 计算深度信息 depth_z。
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    # 初始化相机坐标系下的点云
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)  # 利用像素坐标和深度信息初始化相机坐标系下的点云

    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    # 根据指定的方法计算均方距离initializing the scale of the Gaussians。
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian ** 2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    # 将点云与颜色信息结合，形成彩色的点云。
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    # 是一个张量（tensor），包含了彩色的点云数据。它的格式是一个二维张量，形状为 (N, 6)，其中 N 是点云中点的数量。每一行代表一个点，包含了点的三维坐标（x、y、z）以及颜色信息（R、G、B）

    # Select points based on mask
    # 如果提供了掩码 mask，则基于掩码选择特定的点
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }  # params for 3d gaussian

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    # 初始化单个高斯轨迹来模拟相机相对于第一帧的姿势
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.

    Args:
        params: dict of parameters 一个包含各种参数的字典
        time_idx: time index to transform to 表示时间索引，用于指定转换到哪一帧。
        gaussians_grad: enable gradients for Gaussians  一个布尔值，表示是否启用高斯分布的梯度。
        camera_grad: enable gradients for camera pose 一个布尔值，表示是否启用相机位姿的梯度。

    Returns:
        transformed_pts: Transformed Centers of Gaussians #返回的高斯中心点的变换
    """
    # Get Frame Camera Pose 获取相机位姿：
    if camera_grad:  # 如果 camera_grad 为 True，则获取未归一化的相机旋转 cam_rot 和相机平移 cam_tran
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:  # 否则，使用 .detach() 方法获取它们的副本，确保梯度不会在这里传播。
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()

    # here w2c=Tcw,trans point to camera coordinate
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran
    # R = rel_w2c[:3, :3].inverse()
    # t = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame 获取世界坐标系下高斯分布中心和归一化旋转：
    if gaussians_grad:  # 如果 gaussians_grad 为 True，则获取高斯分布的中心点 pts(不使用 .detach()，所以 pts 是原始张量，它可能是需要计算梯度的。)
        pts = params['means3D']
    else:  # 。否则，使用 .detach() 方法获取其副本(通过使用 .detach() 方法，确保返回的张量是不需要计算梯度的。这可以防止梯度在这个张量上进行传播。)。
        pts = params['means3D'].detach()

    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame 将中心点和未归一化旋转转换到相机坐标系：
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()  # 构建形状为 (N, 4) 的矩阵 pts4，其中 N 是中心点数量，通过在中心点矩阵的最后一列添加全为1的列得到。
    # .cuda() 表示将张量移动到GPU上，如果GPU可用的话。
    # .float() 将张量的数据类型转换为浮点型。
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    # transformed_pts = pts - t
    # transformed_pts = transformed_pts @ R
    # # 利用相机到世界坐标系的变换矩阵 rel_w2c，将这个矩阵应用于 pts4，并提取结果的前三列，得到转换后的中心点 transformed_pts。
    # # 将 pts4 转置（.T）后，利用相机到世界坐标系的变换矩阵 rel_w2c 将其应用于高斯分布的中心点。
    # # 将结果再次转置，然后取前三列，得到形状为 (N, 3) 的张量 transformed_pts。
    # # 这样得到的 transformed_pts 就是高斯分布中心点在相机坐标系中的转换结果，保留了前三个坐标值。

    return transformed_pts


def add_new_gaussians(params, variables, curr_data, time_idx, sil_thres=0.5, mean_sq_dist_method='projective'):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)  # 将高斯模型转换到frame坐标系下
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)  # 获取深度的渲染变量
    # 通过渲染器 Renderer 得到深度图和轮廓图，其中 depth_sil 包含了深度信息和轮廓信息。
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    # non_presence_sil_mask代表当前帧中未出现的区域？
    non_presence_sil_mask = (silhouette < sil_thres)  # 通过设置阈值 sil_thres（输入参数为0.5），创建一个轮廓图的非存在掩码

    # Check for new foreground objects by using GT depth
    # 利用当前深度图和渲染后的深度图，通过 depth_error 计算深度误差，并生成深度非存在掩码 non_presence_depth_mask。
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50 * depth_error.median())

    # Determine non-presence mask
    # 将轮廓图非存在掩码和深度非存在掩码合并生成整体的非存在掩码 non_presence_mask。
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    # 检测到非存在掩码中有未出现的点时，根据当前帧的数据生成新的高斯分布参数，并将这些参数添加到原有的高斯分布参数中
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        # 获取当前相机的旋转和平移信息:
        curr_cam_rot = torch.nn.functional.normalize(
            params['cam_unnorm_rots'][..., time_idx].detach())  # 获取当前帧的相机未归一化旋转信息。
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()  # 对旋转信息进行归一化。
        # 构建当前帧相机到世界坐标系的变换矩阵:
        curr_w2c = torch.eye(4).cuda().float()  # 创建一个单位矩阵
        # 利用归一化后的旋转信息和当前帧的相机平移信息，更新变换矩阵的旋转和平移部分。
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        # 生成有效深度掩码:
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)  # 生成当前帧的有效深度掩码 valid_depth_mask。
        # 更新非存在掩码:
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(
            -1)  # 将 non_presence_mask 和 valid_depth_mask 进行逐元素与操作，得到更新后的非存在掩码。
        # 获取新的点云和平均平方距离:
        # 利用 get_pointcloud 函数，传入当前帧的图像、深度图、内参、变换矩阵和非存在掩码，生成新的点云 new_pt_cld。同时计算这些新点云到已存在高斯分布的平均平方距离 mean3_sq_dist。
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                                   curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                   mean_sq_dist_method=mean_sq_dist_method)  # 参数文件中定义mean_sq_dist_method为projective
        # 初始化新的高斯分布参数:
        # 利用新的点云和平均平方距离，调用 initialize_new_params 函数生成新的高斯分布参数 new_params。
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        # 将新的高斯分布参数添加到原有参数中:
        for k, v in new_params.items():  # 对于每个键值对 (k, v)，其中 k 是高斯分布参数的键，v 是对应的值，在 params 中将其与新参数 v 拼接，并转换为可梯度的 torch.nn.Parameter 对象。
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        # (更新相关的统计信息)初始化一些统计信息，如梯度累积、分母、最大2D半径等。
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        # (更新时间步信息)将新的点云对应的时间步信息 new_timestep（都是当前帧的时间步）拼接到原有的时间步信息中。
        new_timestep = time_idx * torch.ones(new_pt_cld.shape[0], device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep), dim=0)

    # 将更新后的模型参数 params 和相关的统计信息 variables 返回。
    return params, variables


def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]  # 点云
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3] #点云对应的位置信息xyz
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]  高斯球的旋转，四元数的未归一化旋转表示，暗示高斯分布没有旋转。
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")  # 透明度，初始化为0
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    # 构建参数字典 params：params 包含了高斯分布的均值 means3D、颜色 rgb_colors、未归一化旋转 unnorm_rotations、不透明度的对数 logit_opacities 以及尺度的对数 log_scales。
    for k, v in params.items():  # 遍历 params 字典，将其值转换为 torch.Tensor 或 torch.nn.Parameter 类型。
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def initialize_camera_pose(params, curr_time_idx, T, forward_prop):  # 参数文件中，forward_prop是true
    # 此用来确保在这个上下文中没有梯度计算。
    with torch.no_grad():
        transform = T.detach()
        # rot = Rotation.from_matrix(np.linalg.inv(transform[:, :3]))
        rot_quat = rotation_matrix_to_quaternion(transform[:3, :3])
        params['cam_unnorm_rots'][..., curr_time_idx] = rot_quat
        params['cam_trans'][..., curr_time_idx] = transform[:3, 3].T

    return params
