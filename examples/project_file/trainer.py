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
import imageio
import numpy as np
from skimage import img_as_ubyte
import open3d as o3d
from utils import compute_dynamic_position, parse_tapir_track_info

from arap_utils import cal_connectivity_from_points, cal_arap_error, cal_arap_reg
import progression_utils as p_utils
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from tqdm.auto import tqdm


import msplat
import sys
        
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
        
        local_progression_steps: int = 1000
    
        
        
        

    cfg: Config

    
    def __init__(self, cfg: Config, exp_dir: P, name: str, h, w, dataset, init_pcd=None) -> None:
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
        self.dataset = dataset
        
            
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
        self.construct_learnable_motion_param(requires_grad=False)
        self.init_pcd = None
        self.call_hook('init_progress_bar')
        
        # progression
        self.k = torch.Tensor(np.array([[1098.990966796875, 0.0, 400.0], [0.0, 1098.990966796875, 400.0], [0.0, 0.0, 1.0]])).float().to(self.device)
        
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
        mask = batch['mask1'].to(self.device)
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
            
        pass
    
    def train_init_RGB(self, batch):
        
        # freeze color features before training
        # self.model.point_cloud.features.requires_grad = False
        # self.model.point_cloud.features_rest.requires_grad = False
        # self.model.point_cloud.features_rest.opacity = False
        # self.model.point_cloud.features_rest.scale = False
        # self.model.point_cloud.position.requires_grad = False
        # self.model.point_cloud.scaling.requires_grad = False
        # self.model.point_cloud.opacity.requires_grad = False
        # self.model.point_cloud.rotation.requires_grad = False
        # render_features = ['rgb', 'depth', 'opacity']
        render_features = ['rgb']
        self.init_prune()
        intr = torch.Tensor([self.k[0, 0], self.k[1, 1], self.k[0, -1], self.k[1, -1]]).to(self.k)
        # self.call_hook('before_init_train')
        for i in tqdm(range(self.cfg.pose_free.geo_steps)):
            render_results = self.model(batch, render_features=render_features, intrinsic=intr.unsqueeze(0))
            # opa = render_results['opacity']
            # opa_pil = tvF.to_pil_image(opa.squeeze(0))
            # opa_pil.save(self.debug_path / 'opacity.png')
            # self.loss_dict = self.model.get_loss_dict(render_results, batch)
            self.loss_dict = self.model.get_init_loss_dict(render_results, batch)
            
            # cam_center = torch.Tensor([0, 0, 0]).to(self.k)
            # intr = torch.Tensor([self.k[0, 0], self.k[1, 1], self.k[0, -1], self.k[1, -1]]).to(self.k)
            # extr=torch.eye(4).to(intr)
            # opacity = self.model.point_cloud.get_opacity
            # scaling = self.model.point_cloud.get_scaling
            # rotation = self.model.point_cloud.get_rotation
            # shs = self.model.point_cloud.get_shs
            self.loss_dict['loss'].backward()
            loss_value = self.loss_dict['loss'].item()
            self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                                render_results,
                                                                self.white_bg)
        
            with torch.no_grad():
                # self.controller.f_step(**self.optimizer_dict)
                self.model.point_cloud.position.grad.zero_()
                self.model.point_cloud.scaling.grad.zero_()
                self.model.point_cloud.opacity.grad.zero_()
                self.model.point_cloud.rotation.grad.zero_()
                
                
                self.optimizer.update_model(**self.optimizer_dict)
            self.init_step += 1
            # self.call_hook('after_init_train_iter')
            # if loss_value < 0.07:
            #     break
        # self.call_hook('after_geo_init')
        pred_rgb = render_results['rgb']
        pred_rgb_pil = tvF.to_pil_image(pred_rgb.squeeze(0))
        pred_rgb_pil.save(self.debug_path / 'rgb_init.png')
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
        o3d.io.write_point_cloud(str(fname), pcd)
        
        
    def construct_learnable_motion_param(self, requires_grad=True, k_clusters=None):
        if k_clusters is None:
            k_clusters = self.cfg.pose_free.k_clusters
        init_qua = torch.zeros([k_clusters, 4], device=self.device)
        init_qua[:, 0] = 1
        init_trans = torch.zeros([k_clusters, 3], device=self.device)
        param_dict = {
            'quaternion': torch.nn.Parameter(init_qua, requires_grad=requires_grad),
            'translation': torch.nn.Parameter(init_trans, requires_grad=requires_grad)
        }
        self.motion_list += [param_dict]
        pass
    
    
    def simple_progression_update(self, batch):
        
        '''
        estimate point motion and merge the point cloud from depth estimation only
        
        (uv, depth) = self.model.renderer.project_point(
                position, 
                extrinsic_matrix.cuda(), 
                width, 
                height,
                nearest=0.01)
        '''
        position = self.model.point_cloud.position
        self.construct_learnable_motion_param()
        self.construct_motion_optimizer()
        cur_motion_param = self.motion_list[-1]
        position = self.model.compute_dynamic_position(cur_motion_param)
        # extrinsic = torch.eye(4).to(position)
        # (uv, depth) = self.model.renderer.project_point(
        #     position,
        #     extrinsic,
        #     self.w,
        #     self.h,
        #     nearest=0.01
        # )
        
        # 
        # render_features = ['rgb', 'depth', 'opacity', 'pose']
        # render_results = self.model(batch, render_features=render_features)
        
        # opacity = render_results['opacity']
        # pose = render_results['pose']
        pass
    
    def retrieve_points_given_uv(self):
        pass
    
    def update_motion(self, batch):
        
        # update point motion with flow guidance and rigid constraints
        # point_cloud_feats = ['position', 'rotation', '']
        # pcd = self.model.point_cloud
        # pcd_attrs = self.model.point_cloud.attributes
        # for att in pcd_attrs:
        #     cur_att = getattr(pcd, att['name'])
        #     cur_att.requires_grad_ = False
        render_features = ['rgb', 'flow']
        self.model.point_cloud.eval()
        # kms_one_hot = self.run_kmeans()
        self.construct_learnable_motion_param()
        self.construct_motion_optimizer()
        self.model.run_kmeans(self.cfg.pose_free.k_clusters)
        self.cur_motion_step = 0
        self.call_hook('before_motion_update')
        for i in range(self.cfg.pose_free.motion_steps):
            
            self.motion_optimizer.zero_grad()
            render_results = self.model(batch, render_features=render_features, motion_params=self.motion_list[-1])
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
            next_rgb = render_results['future_dict']['rgb'].detach().squeeze(0).permute(1, 2, 0).cpu()
            self.save_img(next_rgb, self.debug_path / 'rgb2_pred.png')
            pred_coord = np.round(mask_pred_flow)
            rgb1_overlay = draw_points(rgb1, flow_pos[:, :2].astype(np.int16))
            rgb2_overlay_pred = draw_points(rgb2, pred_coord.astype(np.int16))
            rgb2_overlay = draw_points(rgb2, fw_flow.astype(np.int16))
            plt.imsave(str(self.debug_path / 'rgb1_overlay.png'), rgb1_overlay)
            plt.imsave(str(self.debug_path / 'rgb2_overlay.png'), rgb2_overlay)
            plt.imsave(str(self.debug_path / 'rgb2_overlay_pred.png'), rgb2_overlay_pred)
            dynamic_pos = self.model.compute_dynamic_position(self.motion_list[-1]).view(-1, 3).detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dynamic_pos)
            o3d.io.write_point_cloud(str(self.debug_path / 'dynamic_pos.ply'), pcd)
            pass
        
    @staticmethod
    def save_img(img, fname):
        img_np = torch.clamp(img, 0.0, 1.0).numpy()
        imageio.imwrite(str(fname), img_as_ubyte(img_np))
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
        
    def remove_floater_dbscan(self):
        pos = self.model.point_cloud.position.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=1000))
        valid_mask = labels >= 0
        valid_mask = torch.from_numpy(valid_mask).to(self.device)
        self.prune_given_valid_mask(valid_mask)
        pass
    
    # def train_progress(self):
    #     pass
    
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
                
    def train_progress(self, batch):
        self.init_pcd = None
        self.global_step = 0
        self.local_step = 0
        self.call_hook('before_global_progression')
        for i in range(len(self.dataset.img_names)):
            batch = self.dataset.__getitem__(i)
            if self.init_pcd is None:
                self.init_pcd = torch.from_numpy(batch['cur_pts']).to(self.device).float()
                pass
            self.global_progression_iter(batch)
            
            if self.global_step % 10 == 0:
                if self.cfg.pose_free.debug:
                    pcd_name = self.debug_path / f'progression_{i:04d}.ply'
                    self.position_to_ply(pcd_name, self.init_pcd)
                    
            
            self.global_step += 1
            self.call_hook('after_global_progression_iter')
            
        self.call_hook()
        pass
    
    def global_progression_iter(self, batch):
        
        pos1 = torch.from_numpy(batch['flow_pos1']).to(self.device)
        depth1 = torch.from_numpy(batch['depth1']).to(self.device)
        fw_flow = torch.from_numpy(batch['fw_flow']).to(self.device)
        depth2 = torch.from_numpy(batch['depth2']).to(self.device)
        mask2 = torch.from_numpy(batch['mask2']).to(self.device)
        valid_visible, _, confidence = parse_tapir_track_info(fw_flow[..., 2], fw_flow[..., 3])
        
        prev_pts = self.pcd_from_depth_flow(depth1, pos1[:, :2].long())
        next_pts = self.pcd_from_depth_flow(depth2, fw_flow[:, :2])
        
        self.position_to_ply(self.debug_path / 'next_pts.ply', next_pts)
        self.position_to_ply(self.debug_path / 'prev_pts.ply', prev_pts)
        self.position_to_ply(self.debug_path / 'cur_pts.ply', torch.from_numpy(batch['cur_pts']))
        next_pts_mask = self.pcd_from_depth_flow(mask2, fw_flow[:, :2])
        valid_mask = next_pts_mask[:, -1].bool()
        final_valid = torch.logical_and(valid_visible, valid_mask)
        # valid_query = pos1[final_valid, :2]
        # valid_traget = fw_flow[final_valid, :2]
        # prev_pts_valid = self.pcd_from_depth_flow(depth1, valid_query.long())
        # next_pts_valid = self.pcd_from_depth_flow(depth2, valid_traget.long())
        
        prev_pts_valid = prev_pts[final_valid]
        next_pts_valid = next_pts[final_valid]
        # generate flow mask for init_pcd
        cur_motion_mask = torch.zeros_like(self.init_pcd[:, 0]).bool()
        cur_motion_mask = torch.concat([cur_motion_mask, final_valid], dim=0)
        
        # concate points to init_pcd
        self.init_pcd = torch.concat([self.init_pcd, prev_pts], dim=0)
        
        
        # constructe learnable params for init_pcd
        self.construct_learnable_motion_param(k_clusters=self.init_pcd.shape[0])
        self.construct_motion_optimizer()
        
        next_pts_gt = next_pts_valid
        self.call_hook('before_local_progression')
        for i in range(self.cfg.local_progression_steps):
            self.local_progression(cur_motion_mask, next_pts_gt)
            self.local_step += 1 
            self.call_hook('after_local_progression_iter')
        self.call_hook('after_local_progression')
        self.call_hook('after_global_progression_iter')
        pass
    
    def local_progression(self, cur_motion_mask, next_pts_gt):
        self.local_dict = {}
        self.motion_optimizer.zero_grad()
        next_pts_pred = compute_dynamic_position(self.init_pcd, self.motion_list[-1])
        
        # compute flow loss
        # next_pts_gt = next_pts[final_valid]
        next_pts_pred_flow = next_pts_pred[cur_motion_mask]
        abs_flow_loss = torch.nn.functional.l1_loss(next_pts_pred_flow, next_pts_gt)
        
        
        ii, jj, nn, _ = cal_connectivity_from_points(points=self.init_pcd, K=10)
        arap_ip = torch.stack([self.init_pcd, next_pts_pred], dim=0)
        rigid_error = cal_arap_error(arap_ip, ii, jj, nn)
        
        loss = abs_flow_loss + rigid_error
        
        self.local_dict.update(
            {
                'flow_loss': abs_flow_loss,
                'rigid_loss': rigid_error
                }
                               )
        
        
        loss.backward()
        # compute 2D CD
        
        # update motion params
        self.motion_optimizer.step()
        self.motion_scheduler.step()
        
        pass
    
    # def construct_learnable_motion_param(self, requires_grad=True):
    #     init_qua = torch.zeros([self.cfg.pose_free.k_clusters, 4], device=self.device)
    #     init_qua[:, 0] = 1
    #     init_trans = torch.zeros([self.cfg.pose_free.k_clusters, 3], device=self.device)
    #     param_dict = {
    #         'quaternion': torch.nn.Parameter(init_qua, requires_grad=requires_grad),
    #         'translation': torch.nn.Parameter(init_trans, requires_grad=requires_grad)
    #     }
    #     self.motion_list += [param_dict]
    #     pass
    
    
    def pcd_from_depth_flow(self, depth, pos):
        pts_depth = depth[pos.round().long()[:,0], pos.round().long()[:,1]].reshape(-1, 1)
        denorm = torch.from_numpy(np.array([[self.w, self.h]])).to(self.device)
        pos_norm = pos / denorm * 2 - 1
        pts = torch.concat([pos_norm, pts_depth], axis=-1)
        return pts.float()
    
    
    def simple_progression(self, batch):
        
        
        ext = torch.eye(4).to(self.device)
        pts_1 = p_utils.retrieve_point_cloud(batch['depth1'], K=self.k, ext=ext, mask=batch['mask1'])
        pts_2 = p_utils.retrieve_point_cloud(batch['depth2'], K=self.k, ext=ext, mask=batch['mask2'])
        
        self.construct_learnable_motion_param(k_clusters=pts_1.shape[0])
        
        pass
    
    def get_pts(self, batch, fidx=1):
        pass
    
    def flow_motion(self, batch):
        flow_src1 = batch['flow_pos1'].to(self.device)
        flow_dst1 = batch['fw_flow'].to(self.device)
        valid_visible, _, _ = parse_tapir_track_info(flow_dst1[..., 2], flow_dst1[..., 3])
        flow_src_valid = flow_src1[valid_visible]
        flow_dst_valid = flow_dst1[valid_visible]
        
        # flow_pos_mask = 
        depth = batch['depth1'].to(self.device)
        flow_mask = torch.zeros_like(depth)
        flow_idx = flow_src_valid[:, :2].long()
        flow_mask[flow_idx[:, 1], flow_idx[:, 0]] = 1
        # debug_mask = flow_mask.unsqueeze(0)
        # debug_mask_pil = tvF.to_pil_image(debug_mask)
        # debug_mask_pil.save(self.debug_path / 'flow_mask.png')
        
        mask1 = batch['mask1'].to(self.device)
        ext = torch.eye(4).to(self.device)
        
        # world_coords = p_utils.retrieve_point_cloud(depth, self.k, ext).float()
        # world_coords_with_flow = torch.concat([world_coords, flow_mask.view(-1, 1)], dim=-1).float() # [N, 4]
        # masked_indices = torch.where(mask1.view(-1) > 0)
        # pts_with_flow = world_coords_with_flow[masked_indices] #[k, 4]
        
        
        world_coords = self.model.point_cloud.position
        masked_indices = torch.where(mask1.view(-1) > 0)
        selected_flow_mask = flow_mask.view(-1)[masked_indices]
        pts_with_flow = torch.concat([world_coords, selected_flow_mask.view(-1, 1)], dim=-1).float()
        
        # gt_uv_mask = p_utils.retrieve_point_cloud(torch.from_numpy(batch['mask2']).to(pts_with_flow), self.k, ext)[:, :2]
        mask2 = batch['mask2'].to(pts_with_flow)
        mask2_idx = torch.where(mask2 == 1)
        # flow_mask2 = torch.zeros_like(depth)
        # flow_mask2[mask2_idx[0], mask2_idx[1]] = 1
        # debug_mask2 = tvF.to_pil_image(flow_mask2)
        # debug_mask2.save(self.debug_path / 'debug_mask2.png')
        gt_uv_mask = torch.stack([mask2_idx[1], mask2_idx[0]], dim=1).to(pts_with_flow)
        # debug
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_with_flow.cpu().numpy()[:, :3])
        # o3d.io.write_point_cloud(str(self.debug_path / 'mask_pts.ply'), pcd)
        
        # construct optimizer & params
        self.construct_learnable_motion_param(k_clusters=pts_with_flow.shape[0])
        self.construct_motion_optimizer()
        
        # render params
        cam_center = torch.Tensor([0, 0, 0]).to(ext)
        intr = torch.Tensor([self.k[0, 0], self.k[1, 1], self.k[0, -1], self.k[1, -1]]).to(self.k)
        extr=torch.eye(4).to(intr)
        opacity = self.model.point_cloud.get_opacity.detach()
        scaling = self.model.point_cloud.get_scaling.detach()
        rotation = self.model.point_cloud.get_rotation.detach()
        shs = self.model.point_cloud.get_shs.detach()
        
        # scaling, rotation, opacity, shs = self.gaussian_point_init(pts_with_flow[:, :3])
        # optimization loop:
        # for i in tqdm(range(1000)):
        total = self.cfg.pose_free.motion_steps
        with tqdm(total=total, position=0, leave=True) as pbar:
            for i in range(total):
                self.motion_optimizer.zero_grad()
                
                final_pos = compute_dynamic_position(pts_with_flow[:, :3].float(), self.motion_list[-1])
                
                # compute loss
                
                # flow loss
                flow_mask = pts_with_flow[:, -1]
                
                (flow_uv_pred, _ ) = msplat.project_point(
                    final_pos,
                    intr=intr,
                    extr=extr,
                    W=self.w, 
                    H=self.h,
                    nearest=0.2
                )
                
                flow_pos_pred = flow_uv_pred[flow_mask.bool()]
                flow_loss = 10 * torch.nn.functional.l1_loss(flow_pos_pred, flow_dst_valid[:, :2])
                
                # rigid regularization
                arap_reg = 100 * cal_arap_reg(pts_with_flow[:, :3], final_pos, K=10)
                
                # projected 2D Chamfer Distance ?
                # import pytorch3d
                # from pytorch3d.loss import chamfer_distance
                # cd, _ = chamfer_distance(flow_uv_pred.unsqueeze(0), gt_uv_mask.unsqueeze(0), single_directional=True)
                
                # render depth image and get depth supervision
                # def render_iter(self,
                #     height,
                #     width,
                #     extrinsic_matrix,
                #     intrinsic_params,
                #     camera_center,
                #     position,
                #     opacity,
                #     scaling,
                #     rotation,
                #     shs,
                #     **kwargs) -> dict:
                
                render_dict = {
                    'height': self.h,
                    'width': self.w,
                    'extrinsic_matrix': ext,
                    'intrinsic_params': intr,
                    'camera_center': cam_center,
                    'position': final_pos,
                    'rotation': rotation,
                    'opacity': opacity,
                    'scaling': scaling,
                    'shs': shs,
                    'render_features': ['depth', 'opacity', 'rgb']
                }
                
                render_results = self.model.renderer.render_iter(**render_dict)
                # # render_depth = render_results['rendered_features_split']['depth']
                # # # render_depth_pil = tvF.to_pil_image(render_depth)
                # # # render_depth_pil.save(self.debug_path / 'render_depth.png')
                
                # # render_opacity = render_results['rendered_features_split']['opacity']
                # # # opacity_pil = tvF.to_pil_image(render_opacity)
                # # # opacity_pil.save(self.debug_path / 'render_opa.png')
                
                # # gt_depth = batch['depth2'].to(render_depth).unsqueeze(0)
                
                # # abs_depth_loss = 10 * F.l1_loss(render_depth, gt_depth)
                
                render_rgb = render_results['rendered_features_split']['rgb']
                gt_rgb = batch['rgb2'].to(render_rgb).permute(2, 0, 1)
                rgb_loss = 0.1*self.model.compute_rgb_loss(render_rgb, gt_rgb)
                # add up the loss
                # if i > 1500:
                # loss = flow_loss + arap_reg #+ cd
                # loss = flow_loss + arap_reg #+ abs_depth_loss
                loss = flow_loss + arap_reg + rgb_loss
                loss.backward()
                self.motion_optimizer.step()
                self.motion_scheduler.step()
                # postfix = f'loss: {loss.item():.4f}, flow_loss: {flow_loss.item():.4f}, arap_reg: {arap_reg.item():.4f}'
                
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'flow_loss': f'{flow_loss.item():.4f}',
                    # 'cd': f'{0:.2f}'
                    # 'depth_loss': f'{abs_depth_loss:0.4f}'
                    'rgb_loss': f'{rgb_loss.item():.4f}',
                    # 'cd': f'{cd.item():.4f}'
                    'arap_reg': f'{arap_reg.item():.4f}'
                }
                
                pbar.set_postfix(postfix)
                pbar.update(1)
        
        # save output for visualization
        self.position_to_ply(self.debug_path / 'world_coords.ply', pts_with_flow[:, :3])
        self.position_to_ply(self.debug_path/'flow_motion.ply', final_pos)
        
        # suppose we now obtain the correct motion
        # find the correct scale for depth2 and add points to pts1
        
        depth2 = batch['depth2'].to(self.device)
        
        mask2 = batch['mask2'].to(self.device)
        
        render_rgb_pil = tvF.to_pil_image(render_rgb)
        render_rgb_pil.save(self.debug_path / 'pred_rgb_motion.png')
        
        pts_2 = p_utils.retrieve_point_cloud(depth2, self.k, torch.eye(4).to(self.device), mask=mask2).float()
        
        self.position_to_ply(self.debug_path / 'pts2.ply', pts_2)
        
        combine = torch.concat([pts_2, pts_with_flow[:, :3]], dim=0)
        self.position_to_ply(self.debug_path / 'combine_before.ply', combine)
        
        combine = torch.concat([pts_2, final_pos], dim=0)
        self.position_to_ply(self.debug_path / 'combine_after.ply', combine)
        
        # estimate scale based on the optical flow points
        flow_final_pos = final_pos[flow_mask.bool()]
        batch['flow_final_pos'] = flow_final_pos
        batch['flow_final_uv'] = flow_pos_pred
        batch['final_pos'] = final_pos
        batch['pts_with_flow'] = pts_with_flow
        batch['flow_dst_valid'] = flow_dst_valid
        
        return batch
        # pass
        
    @staticmethod
    def gaussian_point_init(position, max_sh_degree=3):
        from pointrix.model.point_cloud.utils.point_utils import k_nearest_sklearn
        num_points = len(position)    
        distances= k_nearest_sklearn(position.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)

        # scales = torch.log(avg_dist).repeat(1, 3).to(position)
        scales = 0.01 * torch.ones_like(position)
        # Efficiently create a batch of identity quaternions
        rots = torch.eye(4)[:1].repeat(num_points, 1).to(position)
        # opacities = sigmoid_inv(opc_init_scale * torch.ones((num_points, 1), dtype=torch.float32))
        opacities = torch.ones((num_points, 1), dtype=torch.float32).to(position)
        features_rest = torch.zeros(
            (num_points, (max_sh_degree+1) ** 2 - 1, 3),
            dtype=torch.float32
        ).to(position)

        return scales, rots, opacities, features_rest
        # pass
    
    def construct_learnable_scale(self):
        
        self.motion_list[-1].update({'scale': torch.nn.Parameter(torch.zeros(1).to(self.device))})
        params = [
            {'params': self.motion_list[-1]['scale'],
             'lr': 1e-3}
        ]
        self.scale_optimizer = torch.optim.SGD(params)
        
        pass
    
    def estimate_scale(self, batch):
        
        self.construct_learnable_scale()
        
        pred_pts = batch['flow_final_pos'].detach()
        
        depth2 = batch['depth2'].to(self.device).float()
        
        pred_uv = batch['flow_final_uv'].detach()
        cur_scale = self.motion_list[-1]['scale']
        self.motion_list[-1]['translation'].requires_grad_ = False
        self.motion_list[-1]['quaternion'].requires_grad_ = False
        
        # from tqdm.auto import tqdm
        total = 5000
        
        with tqdm(total=total, position=0, leave=True) as pbar:
            for i in range(total):
                self.scale_optimizer.zero_grad()
                scaled_depth = depth2 * cur_scale
                target_pts = p_utils.get_point_cloud_given_uv(scaled_depth, pred_uv[:, 0], pred_uv[:, 1], K=self.k)
                
                # loss, _ = chamfer_distance(pred_pts.unsqueeze(0), target_pts.unsqueeze(0))
                loss, _ = chamfer_distance(target_pts.unsqueeze(0), pred_pts.unsqueeze(0))
                
                loss.backward()
                self.scale_optimizer.step()
                postfix = {
                    'loss_cd': f'{loss.item():.4f}'
                }
                pbar.set_postfix(postfix)
                pbar.update(1)
        pass
        print(f'estimated scale is: {cur_scale[0].item()}')
        # visualize result
        mask2 = batch['mask2']
        pts_2_scaled = p_utils.retrieve_point_cloud(depth2*cur_scale, self.k, torch.eye(4).to(self.device), mask=mask2).float()
        final_pos = batch['final_pos']
        combine = torch.concat([final_pos, pts_2_scaled], dim=0)
        self.position_to_ply(self.debug_path/'scale_estimate.ply', combine)
        batch['pts_2_scaled'] = pts_2_scaled.detach()
        return batch
    
    
    
    def motion_estimation_fine_level(self, batch):
        
        # final_pos = batch['final_pos'].detach()
        self.motion_list[-1]['translation'].requires_grad_ = True
        self.motion_list[-1]['quaternion'].requires_grad_ = True
        pts_2_scaled = batch['pts_2_scaled']
        
        self.construct_motion_optimizer()
        pts_with_flow = batch['pts_with_flow'].detach()
        flow_dst_valid = batch['flow_dst_valid']
        
        total = self.cfg.pose_free.motion_steps
        with tqdm(total=total, position=0, leave=True) as pbar:
            for i in range(total):
                self.motion_optimizer.zero_grad()
                
                final_pos = compute_dynamic_position(pts_with_flow[:, :3].float(), self.motion_list[-1])
                
                # compute loss
                
                # flow loss
                flow_mask = pts_with_flow[:, -1]
                intr = torch.Tensor([self.k[0, 0], self.k[1, 1], self.k[0, -1], self.k[1, -1]]).to(self.k)
                (flow_uv_pred, _ ) = msplat.project_point(
                    final_pos,
                    intr=intr,
                    extr=torch.eye(4).to(intr),
                    W=self.w, 
                    H=self.h,
                    nearest=0.2
                )
                
                flow_pos_pred = flow_uv_pred[flow_mask.bool()]
                flow_loss = 10 * torch.nn.functional.l1_loss(flow_pos_pred, flow_dst_valid[:, :2])
                
                # rigid regularization
                arap_reg = 100 * cal_arap_reg(pts_with_flow[:, :3], final_pos, K=20)
                
                # projected 2D Chamfer Distance ?
                # import pytorch3d
                # from pytorch3d.loss import chamfer_distance
                # cd, _ = chamfer_distance(flow_uv_pred.unsqueeze(0), gt_uv_mask.unsqueeze(0), single_directional=True)
                cd, _ = chamfer_distance(final_pos.unsqueeze(0), pts_2_scaled.unsqueeze(0), single_directional=True)
                cd = 100 * cd
                
                # add up the loss
                loss = flow_loss + arap_reg + cd
                
                loss.backward()
                self.motion_optimizer.step()
                self.motion_scheduler.step()
                # postfix = f'loss: {loss.item():.4f}, flow_loss: {flow_loss.item():.4f}, arap_reg: {arap_reg.item():.4f}'
                
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'flow_loss': f'{flow_loss.item():.4f}',
                    'arap_reg': f'{arap_reg.item():.4f}',
                    # 'cd': f'{0:.2f}'
                    'cd': f'{cd.item():.4f}'
                }
                
                pbar.set_postfix(postfix)
                pbar.update(1)
        
        # save result for visualization
        combine = torch.concat([final_pos, pts_2_scaled], dim=0)
        self.position_to_ply(self.debug_path/'fine_motion_estimation.ply', combine)
        
    def motion_estimate_by_rendering(self, batch):
        
        
        self.model.renderer.render_it()
        pass
        
    
    def add_new_region(self, batch):
        pass
    
    def flow_cluster(self, batch):

        pos1 = batch['flow_pos1'].astype(np.float32)
        
        fw_pos1 = batch['fw_flow'].astype(np.float32)
        
        valid_visible, _, confidence = parse_tapir_track_info(torch.from_numpy(fw_pos1[..., 2]), torch.from_numpy(fw_pos1[..., 3]))
        
        valid_pos1 = pos1[valid_visible]
        valid_pos1_flow = fw_pos1[valid_visible]
        
        R, t, mask = p_utils.estimate_pose_ransac(valid_pos1[:, :2], valid_pos1_flow[:, :2], self.k.cpu().numpy().astype(np.float32))
        
        gt = np.array([
                [
                    -0.9510562419891357,
                    -0.1402907371520996,
                    0.2753360867500305,
                    1.1013445854187012
                ],
                [
                    0.3090169131755829,
                    -0.4317704439163208,
                    0.84739750623703,
                    3.389590263366699
                ],
                [
                    0.0,
                    0.8910064101219177,
                    0.45399051904678345,
                    1.8159619569778442
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
        
        print(f'total number of tracked points: {len(mask)}, number of inliners: {mask.sum()}, number of outliners: {len(mask) - mask.sum()}')
        pass