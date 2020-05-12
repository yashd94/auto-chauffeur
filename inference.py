
# Written by Team Crazy Glitch Asians, NYU Deep Learning Spring 2020

import os
import random
import numpy as np
from tqdm import tqdm
import shutil
from IPython.display import HTML

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import Generator, Discriminator
from data_helper import UnlabeledDataset, LabeledDataset
from helper import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0);

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/home/cassava_media_arts/data'
annotation_csv = '/home/cassava_media_arts/data/annotation.csv'

load_checkpoint_path = '/home/cassava_media_arts/training_checkpoints/cgan_road_images_2.pth.tar'
save_checkpoint_path = '/home/cassava_media_arts/training_checkpoints/cgan_road_images_2.pth.tar'

unlabeled_scene_index = np.arange(106)
labeled_scene_index_train = np.arange(106, 130)
labeled_scene_index_test = np.arange(130, 134)

"""
Evaluate trained generator model on test data (Qualitative)
"""

labeled_testset = LabeledDataset(image_folder=data_dir,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index_test,
                                  img_transform=transforms.Compose([transforms.Resize((256,256)),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                                                   ]),
                                  map_transform=transforms.Compose([transforms.ToPILImage(),
                                                                    transforms.Resize(256),
                                                                    transforms.ToTensor()
                                                                   ]),
                                  extra_info=True
                                 )

testloader = torch.utils.data.DataLoader(labeled_testset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn,
                                          pin_memory=True)

losses_g_L1_eval = []
inference_images = []

generator = Generator(in_ch, out_ch, ngf, n_blocks=n_blocks_g, init_gain=0.02).to(device)
discriminator = Discriminator(in_ch + out_ch, ndf=ndf, n_layers=n_layers_d).to(device)

"""
Load saved weights from training.
"""
if os.path.isfile(load_checkpoint_path):
    print("=> loading checkpoint '{}'".format(load_checkpoint_path))
    
    checkpoint = torch.load(load_checkpoint_path)
    start_epoch = checkpoint['epoch']
    losses_g = checkpoint['losses_g']
    losses_d = checkpoint['losses_d']
    intermediate_images = checkpoint['intermediate_images']
    
    generator.load_state_dict(checkpoint['g_state_dict'])
    discriminator.load_state_dict(checkpoint['d_state_dict'])
    optim_G.load_state_dict(checkpoint['optim_g'])
    optim_D.load_state_dict(checkpoint['optim_d'])
    
    print("=> loaded checkpoint '{}' (epoch {})".format(load_checkpoint_path, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(load_checkpoint_path))



####################################
############ INFERENCE #############
####################################
generator.eval()

for real_18ch, real_map in tqdm(testloader):
    real_18ch = real_18ch.to(device)
    real_map = real_map.to(device)

    """
    GENERATOR INFERENCE
    """
    gen_map = generator(real_18ch)
    loss_g_L1 = criterion_L1(gen_map, real_map)

    """
    COMPILE TRAINING STATISTICS
    """
    losses_g_L1_eval.append(loss_g_L1.item())
    inference_images.append(np.hstack((real_map.cpu().view([1,256,256]), (gen_map.detach().cpu().view([1,256,256])*0.5) + 0.5)))


"""
NOTEBOOK ONLY
"""
# %%capture
# fig = plt.figure(figsize=(4,2))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)).squeeze(), animated=True)] for i in inference_images]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=2000, blit=True)

# HTML(ani.to_jshtml())