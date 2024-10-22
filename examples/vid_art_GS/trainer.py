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

class pseudo_datapipeline:
    point_cloud: None
        
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
        # @dataclass
        
            
        pipeline = pseudo_datapipeline()
        pipeline.point_cloud = init_pcd
        self.model = parse_model(
            self.cfg.model, pipeline, device=self.device)
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
        # self.model.point_cloud.features_rest.opacity = False
        # self.model.point_cloud.features_rest.scale = False
        
        # render_features = ['rgb', 'depth', 'opacity']
        render_features = ['rgb', 'depth', 'opacity', 'pose']
        self.init_prune()
        self.call_hook('before_init_train')
        mask = torch.from_numpy(batch['mask1']).to(self.device)
        bool_mask = mask > 0
        for i in range(self.cfg.pose_free.geo_steps):
            render_results = self.model(batch, render_features=render_features)
            
            # self.loss_dict = self.model.get_loss_dict(render_results, batch)
            self.loss_dict = self.model.get_init_loss_dict(render_results, batch)
            self.loss_dict['loss'].backward()
            loss_value = self.loss_dict['loss'].item()
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
            if loss_value < 0.07:
                break
            # if self.init_step % 200 == 0:
            #     self.prune_given_valid_mask(bool_mask)
        self.call_hook('after_geo_init')
        if self.cfg.pose_free.debug:
            print('finish init training, check results')
            pred_depth = render_results['depth'].view(self.h, self.w).detach().cpu().numpy()
            pred_opacity = render_results['opacity'].view(self.h, self.w).detach().cpu().numpy()
            try:
                pred_rgb = render_results['rgb'].view(3, self.h, self.w).detach().cpu().permute(1, 2, 0).numpy()
                    
                imageio.imwrite(str(self.debug_path / 'debug_rgb.png'), pred_rgb)
            except:
                pass
            import imageio
            imageio.imwrite(str(self.debug_path / 'gt_depth.png'), batch['depth1'])
            imageio.imwrite(str(self.debug_path / 'debug_depth.png'), pred_depth)
            imageio.imwrite(str(self.debug_path / 'debug_opa.png'), pred_opacity)
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
    
    def prune_given_valid_mask(self, valid_mask):
        self.model.point_cloud.remove_points(valid_mask, self.controller.optimizer)
        self.controller.prune_postprocess(valid_mask)
        # pass
    
    
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
        init_qua[:, 0] = 1
        init_trans = torch.zeros([self.cfg.pose_free.k_clusters, 3], device=self.device)
        param_dict = {
            'quaternion': torch.nn.Parameter(init_qua, requires_grad=True),
            'translation': torch.nn.Parameter(init_trans, requires_grad=True)
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
        render_features = ['flow']
        self.model.point_cloud.eval()
        # kms_one_hot = self.run_kmeans()
        self.construct_learnable_motion_param()
        self.construct_motion_optimizer()
        self.model.run_kmeans(self.cfg.pose_free.k_clusters)
        self.cur_motion_step = 0
        self.call_hook('before_motion_update')
        for i in range(self.cfg.pose_free.motion_steps):
            
            self.motion_optimizer.zero_grad()
            render_results = self.model(batch, render_features=render_features, motion_params=self.motion_list[0])
            self.loss_dict = self.model.get_motion_loss_dict(render_results, batch)
            loss = self.loss_dict['loss']
            loss.backward()
            self.motion_optimizer.step()
            self.motion_scheduler.step()
            
            self.cur_motion_step += 1
            self.call_hook('after_motion_update_iter')
        
        if self.cfg.pose_free.debug:
            # visualize the flow result
            from utils import denormalize_coords
            from matplotlib import pyplot as plt
            mask_pred_flow = render_results['mask_pred_flow'].detach().cpu().numpy()
            flow_pos = batch['flow_pos1']
            rgb1 = batch['rgb1']
            rgb2 = batch['rgb2']
            fw_flow = batch['fw_flow'][:, :2]
            from utils import draw_points
            import numpy as np
            # rgb1_overlay = draw_points(rgb1)
            pred_coord = np.round(mask_pred_flow)
            rgb1_overlay = draw_points(rgb1, flow_pos[:, :2].astype(np.int16))
            rgb2_overlay_pred = draw_points(rgb2, pred_coord.astype(np.int16))
            rgb2_overlay = draw_points(rgb2, fw_flow.astype(np.int16))
            plt.imsave(str(self.debug_path / 'rgb1_overlay.png'), rgb1_overlay)
            plt.imsave(str(self.debug_path / 'rgb2_overlay.png'), rgb2_overlay)
            plt.imsave(str(self.debug_path / 'rgb2_overlay_pred.png'), rgb2_overlay_pred)
            pass
        
    def construct_motion_optimizer(self):
        
        init_q_lr = self.cfg.pose_free.init_q_lr
        end_q_lr = self.cfg.pose_free.end_q_lr
        init_t_lr = self.cfg.pose_free.init_t_lr
        end_t_lr = self.cfg.pose_free.end_t_lr
        max_steps = self.cfg.pose_free.motion_steps
        
        lambda_q_lr = lambda epoch: (init_q_lr - end_q_lr) * (1 - epoch / max_steps) + end_q_lr
        lambda_t_lr = lambda epoch: (init_t_lr - end_t_lr) * (1 - epoch / max_steps) + end_t_lr
        
        labmda_lrs = [
            lambda_q_lr,
            lambda_t_lr
        ]
        
        motion_params = self.motion_list[-1]
        param_group_list = [
            {
                'params': motion_params['quaternion'],
                'lr': init_q_lr
             },
            {
                'params': motion_params['translation'],
                'lr': init_t_lr
            }
        ]
        self.motion_optimizer = torch.optim.Adam(param_group_list)
        self.motion_scheduler = torch.optim.lr_scheduler.LambdaLR(self.motion_optimizer, lr_lambda=labmda_lrs)
        
        
        
    def run_kmeans(self):
        from torch_kmeans import KMeans
        model = KMeans(n_clusters=self.cfg.pose_free.k_clusters)
        pts = self.model.point_cloud.position.unsqueeze(0)
        label = model(pts)
        label_one_hot = torch.nn.functional.one_hot(label.labels.view(-1), num_classes=self.cfg.pose_free.k_clusters)
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