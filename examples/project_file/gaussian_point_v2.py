from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from pointrix.model.point_cloud.points import PointCloud, POINTSCLOUD_REGISTRY
from torch import nn
from pointrix.model.point_cloud.utils.point_utils import (
    sigmoid_inv,
    k_nearest_sklearn
)

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors



def gaussian_point_init(position, max_sh_degree, opc_init_scale=0.1):
    num_points = len(position)    
    distances= k_nearest_sklearn(position.data, 3)
    distances = torch.from_numpy(distances)
    avg_dist = distances.mean(dim=-1, keepdim=True)

    # scales = torch.log(avg_dist).repeat(1, 3)
    scales = torch.ones_like(position)
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
