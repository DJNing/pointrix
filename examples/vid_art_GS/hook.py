import os

from pointrix.hook.log_hook import LogHook
from pointrix.hook.base_hook import HOOK_REGISTRY
from pointrix.utils.visualize import visualize_depth
from pointrix.logger.writer import ProgressLogger
# from trainer import ArtVidTrainer

@HOOK_REGISTRY.register()
class ArtVidLogHook(LogHook):
    def before_init_train(self, trainner) -> None:
        """
        some operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        self.init_step = 0
        self.progress_bar = ProgressLogger(description='training', suffix='iter/s')
        self.progress_bar.add_task("geo_init", "Training Progress for geometry initialization", trainner.cfg.pose_free.geo_steps, log_dict={})
        self.progress_bar.add_task("motion_update", "Training Progress for motion estimation", trainner.cfg.pose_free.motion_steps, log_dict={})
        # self.progress_bar.add_task("train", "Training Progress", trainner.cfg.max_steps, log_dict={})
        # self.progress_bar.add_task("validation", "Validation Progress", len(trainner.datapipeline.validation_dataset), log_dict={})
        # self.progress_bar.reset("validation", visible=False)
        self.progress_bar.reset('motion_update', visible=False)
        self.progress_bar.start()
    
    def after_init_train_iter(self, trainner) -> None:
        for param_group in trainner.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud." + "position":
                pos_lr = param_group['lr']
                break

        log_dict = {
            "num_pt": len(trainner.model.point_cloud),
            "pos_lr": pos_lr
        }
        log_dict.update(trainner.loss_dict)

        for key, value in log_dict.items():
            if 'loss' in key:
                # self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{value:.{7}f}"})

            # if trainner.writer and key != "optimizer_params":
            #     trainner.writer.write_scalar(key, value, trainner.global_step)

        if trainner.init_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pts": f"{len(trainner.model.point_cloud)}",
            })
            self.progress_bar.update("geo_init", step=trainner.cfg.bar_upd_interval, log=self.bar_info)
        pass
    
    def after_geo_init(self, trainner) -> None:
        self.progress_bar.reset('geo_init', visible=False)
        pass
    
    def before_motion_update(self, trainner) -> None:
        self.progress_bar.reset('motion_update', visible=True)
        self.bar_info = {}
    
    def after_motion_update_iter(self, trainner) -> None:
        for param_group in trainner.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud." + "position":
                pos_lr = param_group['lr']
                break

        log_dict = {
            "num_pt": len(trainner.model.point_cloud),
            # "pos_lr": pos_lr
            
        }
        log_dict.update(trainner.loss_dict)

        for key, value in log_dict.items():
            if 'loss' in key:
                # self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{value:.{7}f}"})

            # if trainner.writer and key != "optimizer_params":
            #     trainner.writer.write_scalar(key, value, trainner.global_step)

        if trainner.cur_motion_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pts": f"{len(trainner.model.point_cloud)}",
            })
            self.progress_bar.update("motion_update", step=trainner.cfg.bar_upd_interval, log=self.bar_info)
        pass
    
    # def after_val_iter(self, trainner) -> None:
    #     self.progress_bar.update("validation", step=1)
    #     for key, value in trainner.metric_dict.items():
    #         if key in self.losses_test:
    #             self.losses_test[key] += value

    #     image_name = os.path.basename(trainner.metric_dict['rgb_file_name'])
    #     iteration = trainner.global_step
    #     if 'depth' in trainner.metric_dict:
    #         visual_depth = visualize_depth(trainner.metric_dict['depth'].squeeze(), tensorboard=True)
    #         trainner.writer.write_image(
    #         "test" + f"_view_{image_name}/depth",
    #         visual_depth, step=iteration)
    #     trainner.writer.write_image(
    #         "test" + f"_view_{image_name}/render",
    #         trainner.metric_dict['images'].squeeze(),
    #         step=iteration)

    #     trainner.writer.write_image(
    #         "test" + f"_view_{image_name}/ground_truth",
    #         trainner.metric_dict['gt_images'].squeeze(),
    #         step=iteration)
        
    #     trainner.writer.write_image(
    #         "test" + f"_view_{image_name}/normal",
    #         trainner.metric_dict['normal'].squeeze(),
    #         step=iteration)
    #     trainner.writer.write_image(
    #         "test" + f"_view_{image_name}/normal_gt",
    #         trainner.metric_dict['normal_gt'].squeeze(),
    #         step=iteration)