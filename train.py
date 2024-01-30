import os
import utils
import models
from models import WeightNet
import cross_attention
from cross_attention import CrossAttention
import datasets
import torch as t
from torch import nn
import numpy as np
from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader
from configs import config_sh as config
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.cross_checkpoints_dir, exist_ok=True)

    model = Net()

    resume_epoch = 0
    utils.ini_model_params(model, config.ini_params_mode)
    print(
        "Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Initialize the Model Parameters Use {}".format(
            config.ini_params_mode))


    if config.use_gpu:
        device = t.device('cuda')
    model.to(device)



    train_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len, seq_interval=config.seq_interval,
                                   train=True,
                                   test=False, nonzero_points_threshold=None)
    train_dataloader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    valid_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len, seq_interval=config.seq_interval,
                                   train=False,
                                   test=False, nonzero_points_threshold=None)
    valid_dataloader = DataLoader(valid_dataset, config.valid_batch_size, shuffle=False, num_workers=config.num_workers)
    iters_per_epoch = train_dataloader.__len__()
    iters = resume_epoch * iters_per_epoch



    optimizer = t.optim.Adam(student_model.parameters(), lr=config.learning_rate, betas=config.optim_betas)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',#当监控指标停止下降时，衰减学习率

    best_csi = 0
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train")


    for epoch in range(resume_epoch, config.train_max_epochs):
            
            if (iter + 1) % config.loss_log_iters == 0:
                print('Time: ' + datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    (epoch + 1), (iter + 1) * config.train_batch_size, len(train_dataset),
                                 100. * (iter + 1) / iters_per_epoch, loss.item()))


        scheduler.step(loss)


        pod, far, csi = utils.valid(model, valid_dataloader, config, dBZ_threshold= config.dBZ_threshold)
        print('Time: ' + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\tPOD: {:.4f}\tFAR: {:.4f}\tCSI: {:.4f}'.format(
            (epoch + 1), pod, far, csi))

        if csi > best_csi:
            best_csi = csi
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\t  Save the Current Best Model'.format(epoch + 1))
            t.save(student_model.state_dict(), '{}/{}.pth'.format(config.checkpoints_dir, config.model_name))


        if (epoch + 1) % config.model_save_fre == 0:
                t.save(student_model.state_dict(),
                       '{}/{}_epoch_{}.pth'.format(config.checkpoints_dir, config.model_name, (epoch + 1)))


    writer.close()


if __name__ == '__main__':
    train()
