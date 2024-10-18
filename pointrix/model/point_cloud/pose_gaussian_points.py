import torch
import torch.nn as nn
from dataclasses import dataclass

from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from .points import PointCloud, POINTSCLOUD_REGISTRY

@POINTSCLOUD_REGISTRY.register()
class PoseGaussianPointCloud(GaussianPointCloud):
    """
    
    """
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2

    cfg: Config
    
    def setup(self, point_cloud=None):
        # initialize point cloud base on depth map
        self.point_cloud = None
                
        