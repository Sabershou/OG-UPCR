import torch
from torch import nn as nn
from models.block import UnaryBlock
from utils.transformations import transform_points_Rt
from .alignment import align, align_SC2PCR, align_ransac, align_mac, align_IPID
from .backbones import *
from .correspondence import get_correspondences
from .model_util import get_grid, grid_to_pointcloud, points_to_ndc
from .renderer import PointsRenderer
from .recon_helpers import setup_camera
from .gaussian_renderer import initialize_first_timestep, initialize_params, transform_to_frame, \
    transformed_params2rendervar, transformed_params2depthplussilhouette, add_new_gaussians, initialize_camera_pose, \
    initialize_new_timestep
# from monai.networks.nets import UNet
from pytorch3d.ops.knn import knn_points
from models.correspondence import calculate_ratio_test, get_topk_matches
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import warnings

warnings.filterwarnings("ignore")


def project_rgb(pc_0in1_X, rgb_src, renderer):
    # create rgb_features
    B, _, H, W = rgb_src.shape
    rgb_src = rgb_src.view(B, 3, H * W)
    rgb_src = rgb_src.permute(0, 2, 1).contiguous()

    # Rasterize and Blend
    project_0in1 = renderer(pc_0in1_X, rgb_src)

    return project_0in1["feats"]


# baseline
class PCReg(nn.Module):
    def __init__(self, cfg):
        super(PCReg, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False
        self.encode = ResNetEncoder(chan_in, feat_dim, pretrained)
        self.decode = ResNetDecoder(feat_dim, 3, nn.Tanh(), pretrained)

        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.align_cfg = cfg

    def forward(self, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def forward_pcreg(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        vps = []
        cor_loss = []
        for i in range(1, n_views):
            corr_i = get_correspondences(
                P1=pcs_F[0],
                P2=pcs_F[i],
                P1_X=pcs_X[0],
                P2_X=pcs_X[i],
                num_corres=self.num_corres,
                ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
            )
            Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"corres_0{i}"] = corr_i
            output[f"vp_{i}"] = Rt_i

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        return output

    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None

# replace recurrent resblock in baseline with 2 resnet blocks
from .backbones import ResNetEncoder_modified

class Model_3DGS(nn.Module):
    def __init__(self, cfg):
        super(Model_3DGS, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim
        self.sil_thres = 0.5

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2, 5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5, 8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64 * 2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        # self.num_corres = cfg.num_correspodances
        self.num_corres = 2500
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        self.renderer_3DGS = True
        self.ignore_outlier_depth_loss = False
        self.use_sil_mask = False
        self.view_adap = True
        if self.view_adap:
            self.adaptation_a = nn.Parameter(torch.tensor([0.0], requires_grad=True, device='cuda:3'))
            self.adaptation_b = nn.Parameter(torch.tensor([0.0], requires_grad=True, device='cuda:3'))
        else:
            self.adaptation_a = torch.tensor([0.0], device='cuda:3')
            self.adaptation_b = torch.tensor([0.0], device='cuda:3')

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []
        start_time = time.perf_counter()
        # pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        # downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        # downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']
        pcs_F = self.map(feat_i, feat_p, batch)
        output['pcs_F'] = pcs_F
        end_time = time.perf_counter()
        t_feature = end_time - start_time
        output['time_feature'] = torch.tensor([t_feature]).cuda()

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            seeds_index = []
            for i in range(1, n_views):
                start_time = time.perf_counter()
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                end_time = time.perf_counter()
                t_corr = end_time - start_time
                start_time = time.perf_counter()
                # Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
                # Rt_i, cor_loss_i, seeds = align_SC2PCR(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
                Rt_i, cor_loss_i = align_IPID(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
                # Rt_i, cor_loss_i = align_mac(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
                # Rt_i, cor_loss_i = align_ransac(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)
                end_time = time.perf_counter()
                t_align = end_time - start_time
                # seeds_index.append(seeds)
                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)
                output[f'time_corr'] = torch.tensor([t_corr]).cuda()
                # add for visualization
                output[f'time_align'] = torch.tensor([t_align]).cuda()
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
                # output[f"seeds_{i}"] = seeds
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        render_time = 0
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            if self.renderer_3DGS:
                im = []
                dep = []
                depth_mask = []
                for j in range(B):
                    pose_0 = torch.eye(4).cuda().float()
                    # GT = batch['Rt_1']
                    # pose_0 = batch['Rt_0']
                    # pose_0 = batch[f"Rt_0"].to(torch.float32)
                    # pose_0 = pose_0[j, :, :].squeeze(0)
                    if i == 0:
                        color = rgbs[i + 1]
                        depth = batch[f"depth_{i + 1}"]
                        # pose_i = batch[f"gt_pose_{i + 1}"].to(torch.float32)
                        pose_i = torch.eye(4).cuda().float()
                        pose_i[:3, :3] = output[f"vp_{i + 1}"][j, :, :][:3, :3]
                        pose_i[:3, 3] = output[f"vp_{i + 1}"][j, :, :][:3, 3]
                        # pose_i[:3, :3] = GT[j, :, :][:3, :3]
                        # pose_i[:3, 3] = GT[j, :, :][:3, 3]
                        color = color[j, :, :, :].squeeze(0)
                        depth = depth[j, :, :, :]
                        K_batch = K[j, :, :].squeeze(0)
                        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(color, depth,
                                                                                                        K_batch,
                                                                                                        pose_i,
                                                                                                        pose_0,
                                                                                                        n_views)
                        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'intrinsics': K, 'w2c': first_frame_w2c}
                        # T = torch.eye(4).cuda().float()
                        # T[:3, :3] = output[f"vp_{i + 1}"][j, :, :][:3, :3]
                        # T[:3, 3] = output[f"vp_{i + 1}"][j, :, :][:3, 3]
                        # T = T
                        # params = initialize_camera_pose(params, i+1, T, forward_prop=True)
                        transformed_pts = params['means3D']
                        # transformed_pts = transform_to_frame(params, i+1,
                        #                                      gaussians_grad=False,
                        #                                      camera_grad=False)
                        # transformed_pts = transform_points_Rt(params['means3D'].unsqueeze(0), T, inverse=True)
                        # transformed_pts = transformed_pts.squeeze(0)
                    elif i > 0:
                        color = rgbs[0]
                        depth = batch[f"depth_0"]
                        color = color[j, :, :, :].squeeze(0)
                        depth = depth[j, :, :, :]
                        K_batch = K[j, :, :].squeeze(0)
                        params, variables, intrinsics, first_frame_w2c, cam = initialize_new_timestep(color, depth,
                                                                                                        K_batch,
                                                                                                        pose_0,
                                                                                                        pose_i,
                                                                                                        n_views)
                        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'intrinsics': K, 'w2c': first_frame_w2c}
                        transformed_pts = params['means3D']
                        # transformed_pts = transform_to_frame(params, 0,
                        #                                      gaussians_grad=False,
                        #                                      camera_grad=False)
                    rendervar = transformed_params2rendervar(params, transformed_pts)
                    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                                 transformed_pts)
                    rendervar['means2D'].retain_grad()
                    start_time = time.perf_counter()
                    img, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
                    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
                    end_time = time.perf_counter()
                    render_time = render_time + (end_time - start_time)
                    output['render_time'] = torch.tensor([render_time])
                    depth_img = depth_sil[0, :, :].unsqueeze(0)
                    silhouette = depth_sil[1, :, :]
                    presence_sil_mask = (silhouette > self.sil_thres)
                    depth_sq = depth_sil[2, :, :].unsqueeze(0)
                    uncertainty = depth_sq - depth_img ** 2
                    uncertainty = uncertainty.detach()
                    nan_mask = (~torch.isnan(depth_img)) & (~torch.isnan(uncertainty))
                    if self.ignore_outlier_depth_loss:
                        # 如果开启了 ignore_outlier_depth_loss，则基于深度误差生成一个新的掩码 mask，并且该掩码会剔除深度值异常的区域。
                        depth_error = torch.abs(curr_data['depth'] - depth_img) * (curr_data['depth'] > 0)
                        mask = (depth_error < 10 * depth_error.median())
                        mask = mask & (curr_data['depth'] > 0)
                    else:  # 如果没有开启 ignore_outlier_depth_loss，则直接使用深度大于零的区域作为 mask。
                        mask = (curr_data['depth'] > 0)
                    mask = mask & nan_mask
                    if self.use_sil_mask:
                        mask = mask & presence_sil_mask
                    img_adap = (torch.exp(self.adaptation_a)) * img + self.adaptation_b
                    im.append(img_adap)
                    dep.append(depth_img)
                    depth_mask.append(mask.squeeze(0))
                img = torch.stack(im)
                depth_img = torch.stack(dep)
                cover_mask = torch.stack(depth_mask)
                projs.append({"feats": img, "depth": depth_img, "mask": cover_mask})
            else:
                projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C, H * W).permute(1, 0))
            feat_i2p.append(tgt_feat_i[i].reshape(C, H * W).permute(1, 0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  # N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2 * B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i * H * W: (i + 1) * H * W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i * H * W: (i + 1) * H * W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p

    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None

    def initialize_first_timestep(self, dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method,
                                  densify_dataset=None):
        # Get RGB-D Data & Camera Parameters
        # 从数据集中获取第一帧RGB-D数据（颜色、深度）、相机内参和相机姿态。
        color, depth, intrinsics, pose = dataset[0]

        # Process RGB-D Data
        # 将颜色数据调整为PyTorch的形状和范围。
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        # 调整深度数据的形状。
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Process Camera Parameters
        # 提取相机内参并计算相机到世界坐标系的逆矩阵。
        intrinsics = intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose)

        # Setup Camera
        # 使用提取的相机参数设置相机。
        cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

        if densify_dataset is not None:  # 如果提供了密集化数据集，获取第一帧RGB-D数据和相机内参，并进行相应的处理。
            # Get Densification RGB-D Data & Camera Parameters
            color, depth, densify_intrinsics, _ = densify_dataset[0]
            color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
            depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            densify_intrinsics = densify_intrinsics[:3, :3]
            densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                       w2c.detach().cpu().numpy())
        else:
            densify_intrinsics = intrinsics

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 0)  # Mask out invalid depth values
        mask = mask.reshape(-1)
        # 根据颜色、深度、相机内参、相机到世界坐标系的逆矩阵等信息，使用 get_pointcloud 函数获取初始点云。
        # 通过 mask 过滤掉无效深度值。
        init_pt_cld, mean3_sq_dist = self.get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                         mask=mask, compute_mean_sq_dist=True,
                                                         mean_sq_dist_method=mean_sq_dist_method)  # init_pt_cld是对每个点都生成的点云吗

        # Initialize Parameters
        # 利用初始点云和其他信息，使用 initialize_params 函数初始化模型参数和变量
        # params里是生成点云的点位置颜色还有mean3等,variable则是又点云中的点生成的3d高斯性质,如2d半径等
        params, variables = self.initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        # 估计场景半径，用于后续的高斯光斑密集化。
        variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

        if densify_dataset is not None:
            return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
        else:
            return params, variables, intrinsics, w2c, cam

    def get_pointcloud(self, color, depth, intrinsics, w2c, transform_pts=True,
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

        # 如果 transform_pts 为 True（默认为true且没有传入参数），则进行坐标变换，将点云从相机坐标系变换到世界坐标系。
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
        point_cld = torch.cat((pts, cols),
                              -1)  # 是一个张量（tensor），包含了彩色的点云数据。它的格式是一个二维张量，形状为 (N, 6)，其中 N 是点云中点的数量。每一行代表一个点，包含了点的三维坐标（x、y、z）以及颜色信息（R、G、B）

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

    def initialize_params(self, init_pt_cld, num_frames, mean3_sq_dist):
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
