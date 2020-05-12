"""
Written by Team Crazy Glitch Asians, NYU Deep Learning Spring 2020
-------------------------------------------------------------------
This script trains the ResNet-based generator from the segmention GAN
to reconstruct each image sample. To use this network for segmentation
map creation, the final, fully convolutional layer is replaced with a
new layer having a single output channel. 
"""

import os
import random
import sys

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm 
import torchvision.utils as vutils

from models import Generator
from helper import init_weights, save_checkpoint, print_train_stats
from losses import GANLoss
        

from data_helper import UnlabeledDataset, LabeledDataset, LabeledDataset_task2
from helper import collate_fn, draw_filled_box

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled (80,136 unlabeled images)
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled (20,412 labeled images)
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

image_folder = 'data'
annotation_csv = 'data/annotation.csv'

labeled_scene_index = np.arange(106, 134)
img_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5,), std=(0.5,))])

map_transform = transforms.Compose([transforms.Resize(256),
                                  transforms.ToTensor()])

transform = torchvision.transforms.ToTensor()

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, 
                                      scene_index=labeled_scene_index, 
                                      first_dim='sample', 
                                      transform=img_transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=16, 
                                          shuffle=True, num_workers=2)
      

update_stats_rate = 25
debug_mem_usage = False
losses_g = []
losses_d = []
intermediate_images = []
save_checkpoint_path = 'pretrained_recons/PreTrained_gen_ckpt.pth'

in_ch = 18
out_ch = 18
ngf = 64
ndf = 64
n_blocks_g = 7

start_epoch = 0
n_epochs = 200
lr_g = 0.0001
beta1 = 0.5
beta2 = 0.999

real_label = 0.95
gen_label = 0.0

device = torch.device("cuda")

generator = Generator(in_ch, out_ch, ngf, n_blocks=n_blocks_g, init_gain=0.02).to(device)
generator.apply(init_weights)
generator.train()

criterion_reconstruction = nn.MSELoss() 
optim_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))

####################################
########## TRAINING LOOP ###########
####################################

n_batches = len(trainloader)
for epoch in range(start_epoch, n_epochs):
    iters = 1
    for image in tqdm(trainloader):
        image = image.cuda()
        gen_map = generator(image)
        
        loss_g_total = criterion_reconstruction(gen_map, image)
        generator.zero_grad()
        loss_g_total.backward()
        optim_G.step()
        
        """
        COMPILE TRAINING STATISTICS
        """

        losses_g.append(loss_g_total.item())
        
        if iters % update_stats_rate == 1:
            print("Epoch [{}/{}] | Batch [{}/{}] | Loss: {}".format(epoch, n_epochs, iters, n_batches, loss_g_total))

        iters += 1
    
    # Add generated image to intermediate images list.
    with torch.no_grad():
        gen_map = generator(image).detach().cpu()
    intermediate_images.append(vutils.make_grid(gen_map, normalize=True))
    
    save_checkpoint(save_checkpoint_path,
                    {'epoch': epoch + 1,
                     'g_state_dict': generator.state_dict(),
                     'optim_g' : optim_G.state_dict(),
                     'losses_g' : losses_g,
                     'intermediate_images' : intermediate_images,                     
                    })

