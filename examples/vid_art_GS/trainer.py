from pathlib import Path as P
from dataclasses import dataclass, field
from typing import Optional, List

import torch
# from ..utils.config import parse_structured
# from ..optimizer import parse_optimizer, parse_scheduler
# from ..model import parse_model
# from ..controller.gs import DensificationController
from pointrix.hook import parse_hooks
from pointrix.utils.config import parse_structured
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.model import parse_model
from pointrix.controller.gs import DensificationController
from hook import ArtVidLogHook

class ArtVidTrainer():
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
        exporter: dict = field(default_factory=dict)
        controller: dict = field(default_factory=dict)
        
        # Dataset
        dataset_name: str = "NeRFDataset"
        datapipeline: dict = field(default_factory=dict)

        # Device
        device: str = "cuda"

        # Test config
        training: bool = True
        test_model_path: str = ""

        # Training config
        batch_size: int = 1
        num_workers: int = 0
        max_steps: int = 30000
        val_interval: int = 2000
        spatial_lr_scale: bool = True

        # Progress bar
        bar_upd_interval: int = 10
        # Output path
        output_path: str = "output"
        
        # pose free training:
        pose_free: dict = field(default_factory=dict)
        

    cfg: Config

    def __init__(self, cfg: Config, exp_dir: P, name: str, h, w, init_pcd=None) -> None:
        # super().__init__()
        self.exp_dir = exp_dir
        self.start_steps = 1
        self.global_step = 0
        # build config
        self.cfg = parse_structured(self.Config, cfg)
        self.device = self.cfg.device
        self.h = h
        self.w = w
        self.hooks = parse_hooks(self.cfg.hooks)
        # build point cloud model
        self.white_bg = False
        # self.hooks = parse_hooks(self.cfg.hooks)
        # prepare model
        self.model = parse_model(
            self.cfg.model, init_pcd, device=self.device)
        self.model.h = self.h
        self.model.w = self.w
        self.model.construct_train_cam()

        if self.cfg.training:
            # pseudo cameras_extent
            self.setup_for_training()
        if self.cfg.pose_free.debug:
            self.debug_path = P(self.exp_dir) / 'debug'
            self.debug_path.mkdir(exist_ok=True)
        self.init_step = 0
         
        # motion optimization
        self.motion_list = []
        
        
    def setup(self):
        pass
    
    def train_loop(self):
        pass
    
    def train_init(self, batch):
        
        # freeze color features before training
        # self.model.point_cloud.features.requires_grad = False
        # self.model.point_cloud.features_rest.requires_grad = False
        
        render_features = ['rgb', 'depth', 'opacity']
        self.init_prune()
        self.call_hook('before_init_train')
        for i in range(self.cfg.pose_free.geo_steps):
            render_results = self.model(batch, render_features=render_features)
            
            # self.loss_dict = self.model.get_loss_dict(render_results, batch)
            self.loss_dict = self.model.get_init_loss_dict(render_results, batch)
            self.loss_dict['loss'].backward()
            # structure of optimizer_dict: {}
            # example of optimizer_dict = {
            #   "loss": loss,
            #   "uv_points": uv_points,
            #   "visibility": visibility,
            #   "radii": radii,
            #   "white_bg": white_bg
            self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                                render_results,
                                                                self.white_bg)
        
            with torch.no_grad():
                # mask grad
                # self.model.point_cloud.features.grad_ *= 0
                # self.model.point_cloud.features_rest.grad_ *= 0
                self.controller.f_step(**self.optimizer_dict)
                self.optimizer.update_model(**self.optimizer_dict)
            self.init_step += 1
            self.call_hook('after_init_train_iter')
            
        if self.cfg.pose_free.debug:
            print('finish init training, check results')
            pred_depth = render_results['depth'].view(self.h, self.w).detach().cpu().numpy()
            pred_opacity = render_results['opacity'].view(self.h, self.w).detach().cpu().numpy()
            pred_rgb = render_results['rgb'].view(3, self.h, self.w).detach().cpu().permute(1, 2, 0).numpy()
            import imageio
            imageio.imwrite(str(self.debug_path / 'gt_depth.png'), batch['depth1'])
            imageio.imwrite(str(self.debug_path / 'debug_depth.png'), pred_depth)
            imageio.imwrite(str(self.debug_path / 'debug_opa.png'), pred_opacity)
            imageio.imwrite(str(self.debug_path / 'debug_rgb.png'), pred_rgb)
            self.position_to_ply(str(self.debug_path / 'init_pcd.ply'))
            
            
        pass
    
    def init_prune(self):
        # only save points within the depth map
        pos = self.model.point_cloud.position
        pos_depth = pos[:, 2]
        valid_mask = pos_depth > 0
        self.model.point_cloud.remove_points(valid_mask, self.controller.optimizer)
        self.controller.prune_postprocess(valid_mask)
        pass
    
    
    def position_to_ply(self, fname, tensor=None):
        if tensor is None:
            pts = self.model.point_cloud.position.detach().cpu().numpy()
        else:
            pts = tensor.detach().cpu().numpy()
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(fname, pcd)
        
    def construct_learnable_motion_param(self):
        init_qua = torch.zeros([self.cfg.pose_free.k_clusters, 4], device=self.device)
        init_trans = torch.zeros([self.cfg.pose_free.k_clusters, 3], device=self.device)
        param_dict = {
            'quaternion': torch.nn.Parameter(init_qua),
            'translation': torch.nn.Parameter(init_trans)
        }
        self.motion_list += [param_dict]
        pass
    
    def update_motion(self, batch):
        
        # update point motion with flow guidance and rigid constraints
        # point_cloud_feats = ['position', 'rotation', '']
        # pcd = self.model.point_cloud
        # pcd_attrs = self.model.point_cloud.attributes
        # for att in pcd_attrs:
        #     cur_att = getattr(pcd, att['name'])
        #     cur_att.requires_grad_ = False
        
        self.model.point_cloud.eval()
        kms_one_hot = self.run_kmeans()
        self.construct_learnable_motion_param()
        for i in self.cfg.pose_free.motion_steps:
            pass
        
    def init_motion_optimizer(self):
        self.motion_optimizer = None
        self.motion_scheduler = None
        pass
        
        
    def run_kmeans(self):
        from torch_kmeans import KMeans
        model = KMeans(n_clusters=self.cfg.pose_free.k_clusters)
        pts = self.model.point_cloud.position
        label = model(pts)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.cfg.pose_free.k_clusters)
        return label_one_hot

    
    def update_geometry(self, batch):
        pass
    
    def point_cluster(self):
        pass
    
    def setup_for_training(self):
        cameras_extent = 5
        self.schedulers = parse_scheduler(self.cfg.scheduler,
                                            cameras_extent if self.cfg.spatial_lr_scale else 1.
                                            )
        self.optimizer = parse_optimizer(self.cfg.optimizer,
                                            self.model, datapipeline=None,
                                            cameras_extent=cameras_extent)

        self.controller = DensificationController(
            self.cfg.controller, self.optimizer, self.model, cameras_extent=cameras_extent)
        
    
    def train_progress(self):
        pass
    
    def train_global_opt(self):
        pass
    
    @torch.no_grad()
    def validation(self):
        pass
    
    @torch.no_grad()
    def test(self):
        pass
    
    def get_batch_dict(self):
        
        pass
    
    def call_hook(self, fn_name: str, **kwargs) -> None:
        """
        Call the hook method.

        Parameters
        ----------
        fn_name : str
            The hook method name.
        kwargs : dict
            The keyword arguments.
        """
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None