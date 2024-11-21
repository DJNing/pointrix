import os
import argparse
import os
import sys
import warnings
from pointrix.utils.config import load_config
# from pointrix.engine.default_trainer import DefaultTrainer
# from pointrix.engine.art_vid_trainer import ArtVidTrainer
from trainer import ArtVidTrainer
from pointrix.logger.writer import logproject, Logger

# from dataset import ColmapDepthNormalDataset
from model import VidArtModel
from renderer import MsplatOrthoRender
from dataset import GSSimpleDataset, PoseFreeGSDataset

def main(args, extras) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config, cli_args=extras)
    project_path = os.path.dirname(os.path.abspath(__file__))

    logproject(project_path, os.path.join(cfg.exp_dir, 'project_file'), ['py', 'yaml'])

    # initialize custom dataset
    dataset = PoseFreeGSDataset(cfg.trainer.datapipeline.dataset)
    
    # try:
    # cfg.h = dataset.h
    # cfg.w = dataset.w
    
    # init_idx = dataset.find_largest_mask()
    init_idx = 0
    
    init_pcd = dataset.get_dense_init_pcd(index=init_idx)
    gaussian_trainer = ArtVidTrainer(
                        cfg.trainer,
                        cfg.exp_dir,
                        cfg.name,
                        dataset.h,
                        dataset.w,
                        dataset=dataset,
                        init_pcd=init_pcd
                        )
    f_idx = 0
    
    cur_debug_path = f'{f_idx:04d}'
    gaussian_trainer.debug_path = gaussian_trainer.debug_path / cur_debug_path

    gaussian_trainer.debug_path.mkdir(exist_ok=True)    
    while f_idx < len(dataset):
        batch = dataset.__getitem__(f_idx)
        gaussian_trainer.train_init_RGB(batch)
        new_batch = gaussian_trainer.flow_motion(batch)
        gaussian_trainer.optimize_new_frame(new_batch)
        f_idx += int(cfg.trainer.pose_free.frame_interval)
        cur_debug_path = f'{f_idx:04d}'
        gaussian_trainer.debug_path = gaussian_trainer.debug_path.parent / cur_debug_path

        gaussian_trainer.debug_path.mkdir(exist_ok=True)
    # scale_batch = gaussian_trainer.estimate_scale(new_batch)
        # gaussian_trainer.motion_estimation_fine_level(scale_batch)
        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    args, extras = parser.parse_known_args()

    main(args, extras)
