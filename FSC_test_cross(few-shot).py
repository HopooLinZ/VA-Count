import argparse
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm
from util.FSC147 import transform_train, transform_val
from tqdm import tqdm
assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check

import util.misc as misc
import models_mae_cross


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/FSC147/', type=str,
                        help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_positive.json', type=str,
                        help='annotation json file')
    parser.add_argument('--anno_file_negative', default='./data/FSC147/annotation_FSC147_neg2.json', type=str,
                        help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--output_dir', default='./Image',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./output_fim6_dir/checkpoint-0.pth',
                        help='resume from checkpoint')
    parser.add_argument('--external', action='store_true',
                        help='Set this param for using external exemplars')
    parser.add_argument('--box_bound', default=-1, type=int,
                        help='The max number of exemplars to be considered')
    parser.add_argument('--split', default="test", type=str)

    # Training parameters
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--normalization', default=True, help='Set to False to disable test-time normalization')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

os.environ["CUDA_LAUNCH_BLOCKING"] = '5'

class TestData(Dataset):
    def __init__(self, args, split='val', do_aug=True):
        with open(data_path/args.anno_file) as f:
            annotations = json.load(f)
                # Load negative annotations
        with open(args.anno_file_negative) as f:
            neg_annotations = json.load(f)
        with open(data_path/args.data_split_file) as f:
            data_split = json.load(f)

        self.img = data_split[split]
        random.shuffle(self.img)
        self.split = split
        self.img_dir = im_dir
        # self.TransformTrain = transform_train(args, do_aug=do_aug)
        self.TransformVal = transform_val(args)
        self.annotations = annotations
        self.neg_annotations = neg_annotations
        self.im_dir = im_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        # 加载负样本的框
        neg_anno = self.neg_annotations[im_id]  # 假设每个图像ID在负样本注释中都有对应的条目
        neg_bboxes = neg_anno['box_examples_coordinates']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0
            
            rects.append([y1, x1, y2, x2])
        neg_rects = list()
        for neg_bbox in neg_bboxes:
            x1 = neg_bbox[0][0]
            y1 = neg_bbox[0][1]
            x2 = neg_bbox[2][0]
            y2 = neg_bbox[2][1]
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0
            
            neg_rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.load()
        m_flag = 0

        sample = {'image': image, 'lines_boxes': rects,'neg_lines_boxes': neg_rects, 'dots': dots, 'id': im_id, 'm_flag': m_flag}
        sample = self.TransformTrain(sample) if self.split == "train" else self.TransformVal(sample)
        # if self.split == "train":
        #     sample = self.TransformTrain(sample)
        # # print(sample.keys())
        return sample['image'], sample['gt_density'], len(dots), sample['boxes'], sample['neg_boxes'], sample['pos'],sample['m_flag'], im_id
def batched_rmse(predictions, targets, batch_size=100):
    """
    分批计算RMSE
    :param predictions: 模型预测的值，一个PyTorch张量
    :param targets: 真实的值，一个PyTorch张量，与predictions形状相同
    :param batch_size: 每个批次的大小
    :return: RMSE值
    """
    total_mse = 0.0
    total_count = 0

    # 分批处理
    for i in range(0, len(predictions), batch_size):
        batch_predictions = predictions[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # 确保使用float64进行计算以提高精度
        batch_predictions = batch_predictions.double()
        batch_targets = batch_targets.double()

        # 计算批次的MSE
        difference = batch_predictions - batch_targets
        mse = torch.mean(difference ** 2)

        # 累加MSE和计数
        total_mse += mse * len(batch_predictions)
        total_count += len(batch_predictions)

    # 计算平均MSE
    avg_mse = total_mse / total_count

    # 计算RMSE
    rmse_val = torch.sqrt(avg_mse)

    return rmse_val
def batched_mae(predictions, targets, batch_size=100):
    """
    分批计算MAE
    :param predictions: 模型预测的值，一个PyTorch张量
    :param targets: 真实的值，一个PyTorch张量，与predictions形状相同
    :param batch_size: 每个批次的大小
    :return: MAE值
    """
    total_mae = 0.0
    total_count = 0

    # 分批处理
    for i in range(0, len(predictions), batch_size):
        batch_predictions = predictions[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # 计算批次的绝对误差
        absolute_errors = torch.abs(batch_predictions - batch_targets)
        
        # 累加绝对误差和计数
        total_mae += torch.sum(absolute_errors)
        total_count += len(batch_predictions)

    # 计算平均绝对误差
    avg_mae = total_mae / total_count

    return avg_mae

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_test = TestData(external=args.external, box_bound=args.box_bound, split=args.split)
    dataset_test = TestData(args, split='test')
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start testing.")

    # test
    model.eval()

    # some parameters in training
    train_mae = 0
    train_rmse = 0
    train_nae = 0
    tot_load_time = 0
    tot_infer_time = 0

    loss_array = []
    gt_array = []
    pred_arr = []
    name_arr = []
    empties = []
    # val_mae = torch.tensor([0], dtype=torch.float64, device=device)
    # val_mse = torch.tensor([0], dtype=torch.float64, device=device)
    # val_nae = torch.tensor([0], dtype=torch.float64, device=device)
    # val_mae = torch.tensor([0], dtype=torch.long, device=device)
    # val_mse = torch.tensor([0], dtype=torch.float64, device=device)
    # val_nae = torch.tensor([0], dtype=torch.long, device=device)
# 假设 val_mae, val_mse, 和 val_nae 已经在外部初始化为 0

    total_mae = 0.0
    total_mse = 0.0
    total_nae = 0.0
    total_count = 0
    sub_batch_size = 50
    for val_samples, val_gt_density, val_n_ppl, val_boxes,neg_val_boxes, val_pos, _, val_im_names in tqdm(data_loader_test, total=len(data_loader_test), desc="Validation"):
        val_samples = val_samples.to(device, non_blocking=True, dtype=torch.float)  # 使用更高精度
        val_gt_density = val_gt_density.to(device, non_blocking=True, dtype=torch.float)
        val_boxes = val_boxes.to(device, non_blocking=True, dtype=torch.float)
        neg_val_boxes = neg_val_boxes.to(device, non_blocking=True, dtype=torch.float)
        num_samples = val_samples.size(0)
        total_count += num_samples

        for i in range(0, num_samples, sub_batch_size):
            sub_val_samples = val_samples[i:i+sub_batch_size]
            sub_val_gt_density = val_gt_density[i:i+sub_batch_size]

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sub_val_output = model(sub_val_samples, val_boxes[i:i+sub_batch_size], 3)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    neg_sub_val_output = model(sub_val_samples, neg_val_boxes[i:i+sub_batch_size], 3)
                # output = torch.clamp((sub_val_output-neg_sub_val_output),min=0)
                sub_val_pred_cnt = torch.abs(sub_val_output.sum()) / 60
                # sub_val_pred_cnt = torch.abs(output.sum()) / 60
                # neg_sub_val_pred_cnt = torch.abs(neg_sub_val_output.sum()) / 60
                sub_val_gt_cnt = sub_val_gt_density.sum() / 60
                
                sub_val_cnt_err = torch.abs(sub_val_pred_cnt - sub_val_gt_cnt)

                # 逐项添加并检查
                if not torch.isinf(sub_val_cnt_err) and not torch.isnan(sub_val_cnt_err):
                    batch_mae = sub_val_cnt_err.item()
                    batch_mse = sub_val_cnt_err.item() ** 2
                    batch_nae = sub_val_cnt_err.item() / sub_val_gt_cnt.item() if sub_val_gt_cnt.item() != 0 else 0

                    total_mae += batch_mae * sub_val_samples.size(0)
                    total_mse += batch_mse * sub_val_samples.size(0)
                    total_nae += batch_nae * sub_val_samples.size(0)
                sub_val_pred_cnt = (sub_val_pred_cnt).int()
    final_mae = total_mae / total_count
    final_rmse = (total_mse / total_count) ** 0.5
    final_nae = total_nae / total_count

    print(f'MAE: {final_mae}, RMSE: {final_rmse}, NAE: {final_nae}')

   

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # load data
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
