import os
import utils
import models
from models import WeightNet
import cross_attention
from cross_attention import CrossAttention
import datasets
import torch as t
from torch import nn
from tqdm import tqdm
from configs import config_sh as config
from datetime import datetime
from torch.utils.data import DataLoader
from models import WeightNet


def test(test_samples_only=False, dBZ_threshold = config.dBZ_threshold):

    model = Net()
    model.load_state_dict(t.load('{}/{}.pth'.format(config.checkpoints_dir, config.model_name)))

    if config.use_gpu:
        device = t.device('cuda')
        model.to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load Pretrained Model Successfully")



    if not test_samples_only:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test Model on Test Dataset")
        test_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len,
                                              seq_interval=config.seq_interval,
                                              train=False,
                                              test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, config.test_batch_size, shuffle=False, num_workers=config.num_workers)
        pod, far, csi = utils.valid(model,test_dataloader, config, dBZ_threshold=dBZ_threshold,
                                               eval_by_seq=False)
        print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}'
                  .format(t.mean(pod), t.mean(far), t.mean(csi)))
        #utils.save_test_results(test_results_save_dir, pod, far, csi)


if __name__ == '__main__':
    test()
