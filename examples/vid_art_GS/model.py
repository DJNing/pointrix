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
from pointrix.utils.pose import ConcatRT, quat_to_rotmat, apply_quaternion

from utils import parse_tapir_track_info, denormalize_coords, masked_l1_loss

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
    
    # def dynamic_forward(self, batch=None, render=True, iteration=None, render_features=None):
    #     render_dict = self.construct_render_dict()
    #     pass
    
    def forward(self, batch=None, trainint=True, render=True, iteration=None, init=False, render_features=None, motion_params=None) -> dict:
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
        if motion_params is None:
            render_dict = self.construct_render_dict()
        else:
            render_dict = self.construct_dynamic_render_dict(motion_params=motion_params)
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
        
        # position loss, make sure the pixel location is the same as the rendered pixel location 
        if 'pose' in render_results:
            pos_loss = self.get_pos_loss(render_results, batch)
            loss += pos_loss
        else:
            pos_loss = torch.Tensor([0]).to(loss)
        # position_loss = pass
        
        
        if 'rgb' in render_results.keys():
            rgb_loss = self.get_rgb_loss(render_results, batch)
            loss += rgb_loss
        else:
            rgb_loss = torch.Tensor([0]).to(loss)
        return {
            'loss': loss,
            'depth_loss': depth_loss,
            'opacity_loss': opacity_loss,
            'pose_loss': pos_loss,
            'rgb_loss': rgb_loss
            # 'rgb_loss': rgb_loss
        }
    
    def get_pos_loss(self, render_results, batch):
        render_pos = render_results['pose'].permute(0, 2, 3, 1) # [1, h, w, 3]
        pred_pos = denormalize_coords(render_pos[..., :2], self.h, self.w).view(1, -1, 2)
        grid_i, gird_j = torch.meshgrid(torch.arange(self.h), torch.arange(self.w), indexing='ij')
        pos_tensor = torch.stack((grid_i, gird_j), dim=-1).to(pred_pos).view(1, -1, 2)
        mask = torch.from_numpy(batch['mask1']).to(pred_pos).view(1, -1)
        mask_pred = pred_pos[mask>0]
        mask_gt = pos_tensor[mask>0]
        loss = 1e-2 * torch.nn.functional.l1_loss(mask_pred, mask_gt)
        return loss
    
    def get_motion_loss_dict(self, render_results, batch) -> dict:
        flow_pos = torch.from_numpy(batch['flow_pos2']).to(self.device) #[N, 4], with 4 as [u, v, occlusions, confidence]
        bw_flow = torch.from_numpy(batch['bw_flow']).to(self.device)
        valid_visible, _, confidence = parse_tapir_track_info(bw_flow[..., 2], bw_flow[..., 3])
        valid_visible = valid_visible.view(-1)
        confidence = confidence.view(-1)
        
        render_flow = render_results['flow'].permute(0, 2, 3, 1) # [1, h, w, 3]
        pred_bw_flow = denormalize_coords(render_flow[..., :2], self.h, self.w)
        
        pixel_mask = torch.zeros_like(pred_bw_flow[..., 0])
        
        query_pixel = flow_pos[:, :2].to(torch.int64)
        pixel_mask[0, query_pixel[:, 0], query_pixel[:, 1]] = 1
        pixel_mask_flatten = (pixel_mask.reshape(-1, self.h * self.w) > 0.5)
        # pred_bw_flow = pred_bw_flow.view(-1, self.w * self.h, 2)
        # mask_pred_flow = pred_bw_flow[pixel_mask_flatten][valid_visible]
        
        mask_pred_flow = self.collect_valid_pos(render_flow, pixel_mask_flatten, valid_visible)
        mask_gt_flow = bw_flow[valid_visible][..., :2]
        
        flow_loss = masked_l1_loss(
            mask_pred_flow,
            mask_gt_flow,
            mask = confidence[valid_visible],
            quantile=0.98
        ) / max(self.h, self.w)
        
        return flow_loss
    
    def collect_valid_pos(self, pred, pixel_mask, valid_mask):
        pred_bw_flow = denormalize_coords(pred[..., :2], self.h, self.w)
        pred_bw_flow = pred_bw_flow.view(-1, self.w * self.h, 2)
        mask_pred_flow = pred_bw_flow[pixel_mask][valid_mask]
        return mask_pred_flow
    
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
    
    def construct_dynamic_render_dict(self, motion_params):
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
        
        pos = self.compute_dynamic_position(motion_params)
        rot = self.compute_dynamic_rotation(motion_params)
        render_dict = {
            "position": pos,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": rot,
            "shs": self.point_cloud.get_shs,
            # camera params
            "extrinsic_matrix": extrinsic_matrix.to(self.device),
            "intrinsic_params": intrinsic_params.unsqueeze(0).to(self.device),
            "camera_center": camera_center.unsqueeze(0).to(self.device),
            "height": self.h,
            "width": self.w,
            "previous_pos": self.point_cloud.position
        }
        return render_dict
    
    def run_kmeans(self, k_clusters):
        from torch_kmeans import KMeans
        model = KMeans(n_clusters=k_clusters)
        pts = self.point_cloud.position.unsqueeze(0)
        label = model(pts)
        label_one_hot = torch.nn.functional.one_hot(label.labels.view(-1), num_classes=k_clusters).to(self.point_cloud.position)
        # return label_one_hot
        self.point_cloud.register_attribute("kmeans_label", label_one_hot, trainable=False)
        return
    
    def compute_dynamic_position(self, motion_params):
        static_pos = self.point_cloud.position
        dy_q = motion_params['quaternion']
        dy_T = motion_params['translation'].unsqueeze(1)
        pts_num = static_pos.shape[0]
        homo_pos = torch.concat(
            (static_pos, torch.ones([pts_num, 1], device=static_pos.device)),
                                dim=1)
        
        # construct transformation matrix
        dy_rot = quat_to_rotmat(dy_q)
        trans = torch.concat([dy_rot, dy_T], dim=1)
        
        trans_homo = torch.zeros([dy_rot.shape[0], 4, 4]).to(dy_rot)
        trans_homo[:, :, :3] = trans
        trans_homo[:, -1, -1] = 1
        
        # collect the correct transformation based on kmeans clustering
        label = self.point_cloud.kmeans_label
        full_trans = torch.mm(label, trans_homo.view(-1, 16)).view(-1, 4, 4)
        
        # get position after transformation
        dy_pos = torch.bmm(full_trans, homo_pos.unsqueeze(-1)).squeeze(-1)
        # torch.einsum('ijk,kv->')
        final_pos = dy_pos[:, :3] / dy_pos[:, 3:]
        
        return final_pos
    
    def compute_dynamic_rotation(self, motion_params):
        dy_q = motion_params['quaternion']
        label = self.point_cloud.kmeans_label
        cur_q = self.point_cloud.get_rotation
        full_dy_q = torch.mm(label, dy_q)
        new_q = apply_quaternion(cur_q, full_dy_q)
        return new_q
    
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