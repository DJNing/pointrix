import torch
import torch.nn.functional as F
from pointrix.model.loss import l1_loss, ssim, psnr
from pointrix.utils.pose import unitquat_to_rotmat
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
from pointrix.model.renderer import parse_renderer
# from .point_cloud import parse_point_cloud
from pointrix.model.point_cloud import parse_point_cloud
from dataclasses import dataclass, field
from pointrix.model.loss import l1_loss, ssim, psnr, LPIPS

class SimpleCamera():
    def __init__(self, h, w):
        self.image_height = h
        self.image_width = w
        self.intrinsic = torch.tensor([w, h, w/2, h/2])
        self.extrinsic = torch.eye(4)
        self.camera_center = torch.zeros(3)
    
    

@MODEL_REGISTRY.register()
class VidArtModel(BaseModel):
    
    @dataclass
    class Config:
        camera_model: dict = field(default_factory=dict)
        point_cloud: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        loss_coef: dict = field(default_factory=dict)
        lambda_ssim: float = 0.2
    cfg: Config
    
    @dataclass
    class temp:
        point_cloud: None
    
    def setup(self, datapipeline, device='cuda', h=None, w=None):
        
        # initialization point cloud
        self.h = h
        self.w = w
        pseudo_datapipeline = self.temp
        setattr(pseudo_datapipeline, 'point_cloud', None)
        self.point_cloud = parse_point_cloud(self.cfg.point_cloud, pseudo_datapipeline).to(device)
        self.renderer = parse_renderer(self.cfg.renderer, white_bg=False, device=device)
        self.point_cloud.set_prefix_name("point_cloud")
        self.device = device
        self.lpips_func = LPIPS()
        
        
        pass
    def construct_train_cam(self):
        
        # simple camera model for compatibility with density controller
        self.training_camera_model = SimpleCamera(self.h, self.w)
        # initialize pose estimation
    def register_transform(self):
        pass
    
    def forward(self, batch=None, trainint=True, render=True, iteration=None, init=False, render_features=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        if iteration is not None:
            self.renderer.update_sh_degree(iteration)
        if batch is None:
            raise ValueError("Batch data input could not be None!")
            
        # prepare render_dict
        render_dict = self.construct_render_dict()
        if render_features is not None:
            render_dict.update({'render_features': render_features})
        if render:
            render_results = self.renderer.render_batch(render_dict, [batch])
        
            return render_results
        else:
            return render_dict
    
    def get_loss_dict(self, render_results, batch) -> dict:
        
        # _loss = self.get_depth_loss(render_results, batch)
        # _loss = self.get_flow_loss(render_results, batch)
        # _loss = self.get_rgb_loss(render_results, batch)
        # _loss = self.get_opacity_loss(render_results, batch)
        # _loss = self.get_rigid_loss(render_results, batch)
        
        pass
    
    def get_init_loss_dict(self, render_results, batch) -> dict:
        depth_loss = self.get_depth_loss(render_results, batch)
        opacity_loss = self.get_opacity_loss(render_results, batch)
        
        loss = self.cfg.loss_coef.depth * depth_loss + self.cfg.loss_coef.opa * opacity_loss #+ rgb_loss
        
        if 'rgb' in render_results.keys():
            rgb_loss = self.get_rgb_loss(render_results, batch)
            loss += rgb_loss
        return {
            'loss': loss,
            'depth_loss': depth_loss,
            'opacity_loss': opacity_loss
            # 'rgb_loss': rgb_loss
        }
    
    def get_motion_loss_dict(self, render_results, batch) -> dict:
        pass
    
    
    def get_depth_loss(self, render_results, batch):
        pred_depth = render_results['depth']
        gt_depth = torch.from_numpy(batch['depth1']).to(pred_depth)
        mask = torch.from_numpy(batch['mask1']).to(pred_depth).view(-1, 1)
        
        pred_masked = pred_depth.view(-1, 1)[mask > 0]
        gt_masked = gt_depth.view(-1, 1)[mask > 0]
        from utils import depth_loss_dpt
        # depth_loss = F.l1_loss(pred_masked, gt_masked)
        depth_loss = depth_loss_dpt(pred_masked, gt_masked)
        
        # depth_empty_loss = F.mse_loss(pred_depth.view(-1, 1)[mask==0], gt_depth.view(-1, 1)[mask==0])
        return depth_loss
    
    def get_flow_loss(self, render_results, batch):
        pass
    
    def get_rgb_loss(self, render_results, batch):
        pred_rgb = render_results['rgb'].squeeze(0).permute(1, 2, 0)
        gt_rgb = torch.from_numpy(batch['rgb1']).to(pred_rgb)
        rgb_loss = F.l1_loss(pred_rgb, gt_rgb)
        
        ssim_loss = ssim(pred_rgb, gt_rgb)
        final_loss = self.cfg.lambda_ssim * ssim_loss + (1 - self.cfg.lambda_ssim) * rgb_loss
        return final_loss
    
    def get_opacity_loss(self, render_results, batch):
        pred_opa = render_results['opacity']
        gt_opa = torch.from_numpy(batch['mask1']).to(pred_opa)
        opa_loss = F.mse_loss(pred_opa, gt_opa)
        return opa_loss
    
    def get_rigid_loss(self, render_results, batch):
        pass
    # def load_ply(self, path):
    #     pass
    # def get_optimizer_dict(self, loss_dict, render_results, white_bg) -> dict:
    #     pass
    
    def construct_render_dict(self):
        """
        define the fixed camera

        Parameters
        ----------
        h : int
            The height of the image.
        w : int
            The width of the image.
        """
        
        camera_center = torch.zeros(3)
        extrinsic_matrix = torch.eye(4).unsqueeze(0)
        intrinsic_params = torch.tensor([self.w, self.h, self.w/2, self.h/2])
        render_dict = {
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            # camera params
            "extrinsic_matrix": extrinsic_matrix.to(self.device),
            "intrinsic_params": intrinsic_params.unsqueeze(0).to(self.device),
            "camera_center": camera_center.unsqueeze(0).to(self.device),
            "height": self.h,
            "width": self.w
        }
        return render_dict