import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from utils import normalize_coords, gen_grid_np
from pathlib import Path as P

def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class GSSimpleDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = P(args.dataset.data_path)
        self.seq_name = P(args.dataset.seq_name)
        # self.img_dir = os.path.join(self.seq_dir, 'JPEGImages/480p', self.seq_name)
        # import pdb
        # pdb.set_trace()
        # self.img_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/images"
        img_path = str(self.seq_dir / self.seq_name)
        self.img_dir = os.path.join(img_path, 'images')
        self.mask_dir = os.path.join(img_path, 'masks')
        self.flow_dir = os.path.join(img_path, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        if self.args.num_imgs < 0:
            self.num_imgs = len(img_names)
        else:
            self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.base_idx = self.args.base_idx
        self.img_names = img_names[self.base_idx:self.num_imgs+self.base_idx]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        # self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)

        img_name2 = np.random.choice(self.img_names)

        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        mask_1 = imageio.imread(os.path.join(self.mask_dir, img_name1)) / 255.
        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        weights = pair_weight

        gt_rgb1 = torch.from_numpy(img1).float().reshape(-1,3)

        data = {'ids1': id1,
                'ids2': id2,
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'weights': weights,  # [n_pts, 1],
                'fg_mask': mask_1
                }
        return data
    
    def get_consecutive_item(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)



        img_name2 = np.random.choice(self.img_names)
        
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        mask_1 = imageio.imread(os.path.join(self.mask_dir, img_name1)) / 255.
        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        weights = pair_weight

        gt_rgb1 = torch.from_numpy(img1).float().reshape(-1,3)

        data = {'ids1': id1,
                'ids2': id2,
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'weights': weights,  # [n_pts, 1],
                'fg_mask': mask_1
                }
        return data
    
    def get_two_frame(self, idx1, idx2):
        pass
    
    
class PoseFreeGSDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = P(args.data_path)
        self.seq_name = args.seq_name
        # self.img_dir = os.path.join(self.seq_dir, 'JPEGImages/480p', self.seq_name)
        # import pdb
        # pdb.set_trace()
        # self.img_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/images"
        self.img_dir = self.seq_dir / self.seq_name / 'images'
        self.mask_dir = self.seq_dir / self.seq_name / 'masks'
        self.depth_dir = self.seq_dir / self.seq_name / 'aligned_depth_anything_v2'
        self.flow_dir = self.seq_dir / self.seq_name / 'bootstapir'
        
        self.detph_files = sorted([str(f) for f in self.depth_dir.glob('*.npy')])
        
        # self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted([f.name for f in self.img_dir.glob("*.png")])
        if self.args.num_imgs < 0:
            self.num_imgs = len(img_names)
        else:
            self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.base_idx = self.args.base_idx
        self.img_names = img_names[self.base_idx:self.num_imgs+self.base_idx]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        # max_interval = 3
        # self.max_interval = mp.Value('i', max_interval)
        # self.num_pts = self.args.num_pts
        # self.grid = gen_grid_np(self.h, self.w)
        
    # def get_init_pcd(self):
    #     batch = self.__getitem__(0)
    #     depth = batch['depth1']
    #     mask = batch['mask1']
        
    #     pass
        
    def __getitem__(self, index):
        id1 = index % self.num_imgs
        
        if id1 == len(self.img_names) - 1:
            id1 = 0
        
        id2 = id1 + 1
        
        rgb_1 = imageio.imread(str(self.img_dir / self.img_names[id1])) / 255.
        rgb_2 = imageio.imread(str(self.img_dir / self.img_names[id2])) / 255.
        
        mask_1 = imageio.imread(str(self.mask_dir / self.img_names[id1])) / 255.
        mask_2 = imageio.imread(str(self.mask_dir / self.img_names[id2])) / 255.
        
        # load depth
        depth_1 = np.load(self.detph_files[id1])
        depth_2 = np.load(self.detph_files[id2])
        
        # load opt flow
        fw_flow_name = f'{id1:04d}_{id2:04d}.npy'
        bw_flow_name = f'{id2:04d}_{id1:04d}.npy'
        
        fw_flow = np.load(str(self.flow_dir / fw_flow_name))
        bw_flow = np.load(str(self.flow_dir / bw_flow_name))
        
        data = {
            'id1': id1,
            'id2': id2,
            'rgb1': rgb_1,
            'rgb2':rgb_2,
            'mask1': mask_1,
            'mask2': mask_2,
            'depth1': depth_1,
            'depth2': depth_2,
            'fw_flow': fw_flow,
            'bw_flow': bw_flow
        }
        
        return data
