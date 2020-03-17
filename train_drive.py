import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from modules import SOFVSR, optical_flow_warp
import argparse
from data_utils import TrainsetLoader
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from math import exp


# SSIM loss
#from pytorch_msssim import msssim, ssim


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    ret = 1.0 / ret

    if full:
        return ret, cs
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True) #False)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=30000000, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    return parser.parse_args()

def main(cfg):
    use_gpu = cfg.gpu_mode
    net = SOFVSR(cfg.upscale_factor, is_training=True)
    if use_gpu:
        net.cuda()
    cudnn.benchmark = True

    train_set = TrainsetLoader(cfg.trainset_dir, cfg.upscale_factor, cfg.patch_size, cfg.n_iters)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)  #  num_workers=4,

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion_L2 = torch.nn.MSELoss()

    # use SSIM loss 
    #criterion_L2 = ssim()


    #if use_gpu:
     #   criterion_L2 = criterion_L2.cuda()
    milestones = [50000, 100000, 150000, 200000, 250000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    loss_list = []

    for idx_iter, (LR, HR) in enumerate(train_loader):
        scheduler.step()

        LR, HR = Variable(LR), Variable(HR)
        if use_gpu:
            LR = LR.cuda()
            HR = HR.cuda()

        (res_01_L1, res_01_L2, flow_01_L1, flow_01_L2, flow_01_L3), (
            res_21_L1, res_21_L2, flow_21_L1, flow_21_L2, flow_21_L3), SR = net(LR)
        warped_01 = optical_flow_warp(torch.unsqueeze(HR[:, 0, :, :], dim=1), flow_01_L3)
        warped_21 = optical_flow_warp(torch.unsqueeze(HR[:, 2, :, :], dim=1), flow_21_L3)

        # losses
        loss_SR = criterion_L2(SR, torch.unsqueeze(HR[:, 1, :, :], 1))  #criterion_L2
        loss_OFR_1 = 1 * (criterion_L2(warped_01, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_01_L3)) + \
                     0.25 * (torch.mean(res_01_L2 ** 2) + 0.01 * L1_regularization(flow_01_L2)) + \
                     0.125 * (torch.mean(res_01_L1 ** 2) + 0.01 * L1_regularization(flow_01_L1))
        loss_OFR_2 = 1 * (criterion_L2(warped_21, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_21_L3)) + \
                     0.25 * (torch.mean(res_21_L2 ** 2) + 0.01 * L1_regularization(flow_21_L2)) + \
                     0.125 * (torch.mean(res_21_L1 ** 2) + 0.01 * L1_regularization(flow_21_L1))

        '''
        loss_OFR_1 = 1 * (criterion_L2(warped_01, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_01_L3)) + \
                     0.25 * (torch.mean(res_01_L2 ** 2) + 0.01 * L1_regularization(flow_01_L2)) + \
                     0.125 * (torch.mean(res_01_L1 ** 2) + 0.01 * L1_regularization(flow_01_L1))
        loss_OFR_2 = 1 * (criterion_L2(warped_21, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_21_L3)) + \
                     0.25 * (torch.mean(res_21_L2 ** 2) + 0.01 * L1_regularization(flow_21_L2)) + \
                     0.125 * (torch.mean(res_21_L1 ** 2) + 0.01 * L1_regularization(flow_21_L1))

        '''
        loss = loss_SR + 0.01 * (loss_OFR_1 + loss_OFR_2) / 2
        loss_list.append(loss.data.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('iter: ' + str(idx_iter+1))

        # save checkpoint
        if idx_iter % 1000 == 0:
            print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
            torch.save(net.state_dict(), 'log/mse2_BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
            loss_list = []

def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)







