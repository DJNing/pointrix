from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from pointrix.model.point_cloud.points import PointCloud, POINTSCLOUD_REGISTRY
from torch import nn
from pointrix.model.point_cloud.utils.point_utils import (
    sigmoid_inv,
    k_nearest_sklearn,
    get_random_feauture
)
from dataclasses import dataclass
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from progression_utils import retrieve_point_cloud


def gaussian_point_init(position, max_sh_degree, opc_init_scale=0.1):
    num_points = len(position)    
    distances= k_nearest_sklearn(position.data, 3)
    distances = torch.from_numpy(distances)
    avg_dist = distances.mean(dim=-1, keepdim=True)
    median_dist = torch.ones_like(avg_dist) * avg_dist.median()
    scales = torch.log(median_dist).repeat(1, 3)
    # scales = 0.01*torch.ones_like(position)
    # Efficiently create a batch of identity quaternions
    rots = torch.eye(4)[:1].repeat(num_points, 1)  
    # opacities = sigmoid_inv(opc_init_scale * torch.ones((num_points, 1), dtype=torch.float32))
    opacities = torch.ones((num_points, 1), dtype=torch.float32)
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    )

    return scales, rots, opacities, features_rest

@POINTSCLOUD_REGISTRY.register()
class GaussianPointCloud_v2(GaussianPointCloud):
    
    def re_init(self, num_points):
        super().re_init(num_points)
        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )
        self.register_attribute("features_rest", features_rest)
        self.register_attribute("scaling", scales)
        self.register_attribute("rotation", rots)
        self.register_attribute("opacity", opacities)
        
@POINTSCLOUD_REGISTRY.register()
class GaussianPointCloud_scale_depth(GaussianPointCloud):
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2
        
    cfg: Config
    
    def setup(self, point_cloud, depth, K, mask):
        
        self.atributes = []
        self.depth = depth
        self.K = K
        self.mask = mask
        pts = retrieve_point_cloud(depth, K, mask=mask)
        
        position = pts.to(self.device)
        features = get_random_feauture(position.shape[0], self.cfg.initializer.feat_dim)
        # position, features = points_init(self.cfg.initializer, point_cloud)
        self.register_buffer('scale', torch.ones(1).to(self.device))
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': self.cfg.trainable,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        
        if self.cfg.trainable:
            self.position = nn.Parameter(
                position.contiguous().requires_grad_(True)
            )
            self.features = nn.Parameter(
                features.contiguous().requires_grad_(True)
            )

        self.prefix_name = self.cfg.unwarp_prefix + "."
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = sigmoid_inv
        self.rotation_activation = torch.nn.functional.normalize

        # scales, rots, opacities, features_rest = gaussian_point_init(
        #     position=self.position,
        #     max_sh_degree=self.cfg.max_sh_degree,
        # )

        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        self.register_attribute("features_rest", features_rest)
        self.register_attribute("scaling", scales)
        self.register_attribute("rotation", rots)
        self.register_attribute("opacity", opacities)
        pass
    
    @property
    def get_position(self):
        scale_depth = self.scale * self.depth
        pts = retrieve_point_cloud(scale_depth, self.K, mask=self.mask)
        return pts
    
    def re_init(self, num_points):
        super().re_init(num_points)
        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )
        self.register_attribute("features_rest", features_rest)
        self.register_attribute("scaling", scales)
        self.register_attribute("rotation", rots)
        self.register_attribute("opacity", opacities)
