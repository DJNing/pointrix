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
    gaussian_trainer = ArtVidTrainer(
                        cfg.trainer,
                        cfg.exp_dir,
                        cfg.name,
                        dataset.h,
                        dataset.w
                        )
    
    batch_init = dataset.__getitem__(0)
    gaussian_trainer.train_init(batch_init)
    gaussian_trainer.update_motion(batch_init)
    #     if cfg.trainer.training:
    #         gaussian_trainer.train_loop()
    #         model_path = os.path.join(
    #             cfg.exp_dir,
    #             "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    #         )
    #         gaussian_trainer.save_model(path=model_path)
    #         gaussian_trainer.test()
    #     else:
    #         gaussian_trainer.test(cfg.trainer.test_model_path)
    #     Logger.print("\nTraining complete.")
    # except:
    #     Logger.print_exception(show_locals=True)
    #     for hook in gaussian_trainer.hooks:
            # hook.exception()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    args, extras = parser.parse_known_args()

    main(args, extras)
