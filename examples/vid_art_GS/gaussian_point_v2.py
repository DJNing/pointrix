from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from pointrix.model.point_cloud.points import PointCloud, POINTSCLOUD_REGISTRY

@POINTSCLOUD_REGISTRY.register()
class GaussianPointCloud_v2(GaussianPointCloud):
    
    def re_init(self, num_points, depth_map=None):
        
        pass
