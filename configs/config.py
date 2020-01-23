# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import numpy as np

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        num_workers = 6
    else:
        num_workers = 0

    hard_mining = True
    epochs = 300
    lr = 0.1
    gamma = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    steps = [25,50,75,100,125,150,175,200,225,250,275]
    rewards_list = [2.2]
    coverages_list = [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.]

    train_batch_size = 128
    test_batch_size = 200

    arch = None
    pretrain = None
    output_dir = None
    model_dir = None
    log_dir = None

config = Config()
