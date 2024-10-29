import os
import dearpygui.dearpygui as dpg
import numpy as np
import torch
import tqdm
# from pointrix.model.camera.cam_utils import OrbitCamera, MiniCam2, construct_canonical_camera_from_focal, focal2fov
from pointrix.utils.gui_cam_utils import OrbitCamera, MiniCam2, construct_canonical_camera_from_focal, focal2fov
from argparse import ArgumentParser
import sys
from pointrix.utils.config import load_config
from pointrix.engine.default_trainer import DefaultTrainer
import time
from gui_interface import register_dpg
from pointrix.utils.visualize import visualize_depth
import imageio
# from pointrix.model.camera.camera_model import CameraModel
# from pointrix.dataset.utils.dataprior import CameraPrior
from examples.vid_art_GS.trainer import ArtVidTrainer
class GUI:
    def __init__(self, args, extras) -> None:
        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0

        # For UI
        self.visualization_mode = 'RGB'
        self.is_play = False
        self.fix_cam = False

        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.fovy = args.fovy
        self.cam = OrbitCamera(args.W, args.H, 
                               r=args.radius, 
                               fovy=args.fovy,
                               center=[0, 0, 0.0])
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.training = False
        self.video_speed = 1.
        self.current_fid_ratio = 0
        self.record = False
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)

        if self.gui:
            dpg.create_context()
            register_dpg(self)


        #### init trainer ####
        # cfg = load_config(args.config, cli_args=extras)
        # self.gaussian_trainer = DefaultTrainer(
        #     cfg.trainer,
        #     cfg.exp_dir,
        # )
        # self.gaussian_trainer.load_model("image_depth/image_depth/chkpnt201.pth")
        # self.gaussian_trainer.model.to(self.gaussian_trainer.device)

    def vis(self, cam_param):
        pass

    def seed_everything(self):
        print(" !!! set seed here !!!")
        pass

    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10
        
        camera = MiniCam2(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid = 0,
            to_opengl=True
        )

        render_dict = {
            "FovX": camera.FoVx,
            "FovY": camera.FoVy,
        }

        if self.fix_cam:
            focal = focal2fov(3.14/8, self.H)
            camera = construct_canonical_camera_from_focal(self.W, self.H, focal)
            render_dict['FovX'] = camera.fovX
            render_dict['FovY'] = camera.fovY

        render_dict.update({
            "camera": camera,
            "height": int(camera.image_height),
            "width": int(camera.image_width),
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "extrinsic_matrix": camera.extrinsic_matrix,
            "intrinsic_matrix": camera.intrinsic_matrix,
            "camera_center": camera.camera_center,
            "scaling_modifier": 1.0 if self.vis_scale_const is None else self.vis_scale_const,
            "enable_ortho_projection": False
        })


        fid = int(self.current_fid_ratio*self.gaussian_trainer.num_imgs) if self.is_play else (time.time()-self.t0) * self.fps_of_fid * self.video_speed % self.gaussian_trainer.num_imgs
        fid = int(fid)

        atributes_dict = self.gaussian_trainer.get_attributes_dict(frame_idx=fid)
        render_dict.update(atributes_dict)
        image = self.gaussian_trainer.renderer.render_iter(**render_dict)
        if self.visualization_mode.lower() == 'rgb':
            buffer_image = image["rendered_features_split"]['rgb']  # [3, H, W]
            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
        elif self.visualization_mode.lower() == 'depth':
            buffer_image = image["rendered_features_split"]['depth']  # [1, H, W]
            self.buffer_image = np.ascontiguousarray(visualize_depth(buffer_image.squeeze()).astype(np.float32) / 255.)
            # buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            # buffer_image = buffer_image.repeat(3, 1, 1)

        self.need_update = True

        ##### add video record. Use 30 frames and 4 fps, and current camera
        if self.record:
            images = []
            for idx in range(self.gaussian_trainer.num_imgs):
                atributes_dict = self.gaussian_trainer.get_attributes_dict(frame_idx=idx)
                render_dict_copy = render_dict.copy()
                render_dict_copy.update(atributes_dict)
                image = self.gaussian_trainer.renderer.render_iter(**render_dict_copy)
                if self.visualization_mode.lower() == 'rgb':
                    buffer_image = image["rendered_features_split"]['rgb']  # [3, H, W]
                    self.buffer_image = (
                        buffer_image.permute(1, 2, 0)
                        .contiguous()
                        .clamp(0, 1)
                        .contiguous()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                images.append((self.buffer_image*255).astype(np.uint8))
            save_path = os.path.join(self.gaussian_trainer.out_dir, "video_record.mp4")
            imageio.mimwrite(save_path, images, fps=4)
            print(f"Save video to {save_path}")
            self.record = False

            

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)


        buffer_image = self.buffer_image
        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS )")
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--config', type=str, default="config.yaml", help="path to config file")

    args, extras = parser.parse_known_args()
    gui = GUI(args=args, extras=extras)
    if args.gui:
        while dpg.is_dearpygui_running():
            gui.test_step()
            dpg.render_dearpygui_frame()
    else:
        gui.test_step()
  