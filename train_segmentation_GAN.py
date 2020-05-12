########################################################################################################
# This script trains a ResNet-based GAN to generate object masks, i.e. binary images that correpond
# to top-down views of the road with rectangular boxes depicting boxes.
########################################################################################################
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

labeled_trainset = LabeledDataset_task2(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  img_transform = img_transform,
                                  map_transform = map_transform,
                                  extra_info=False)

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size = 16, 
                                          shuffle = True, num_workers = 2, collate_fn = collate_fn)

from models import Generator, Discriminator
from helper import GANLoss, save_checkpoint, print_train_stats, init_weights
    
update_stats_rate = 25
debug_mem_usage = False
losses_g = []
losses_d = []
intermediate_images = []
save_checkpoint_path = 'object_masks/ObjMap_GAN_pretrained_ckpt.pth'

in_ch = 18
out_ch = 18
out_ch_new = (in_ch + 1)
ngf = 64
ndf = 64
n_blocks_g = 7
n_layers_d = 3

start_epoch = 0
n_epochs = 200
lr_g = 0.0001
lr_d = 0.0004
beta1 = 0.5
beta2 = 0.999
L1_lambda = 100

real_label = 0.95
gen_label = 0.0

device = torch.device("cuda")
generator = Generator(in_ch, out_ch, ngf, n_blocks=n_blocks_g, init_gain=0.02).to(device)
discriminator = Discriminator((in_ch + 1), ndf=ndf, n_layers=n_layers_d).to(device)

# Load pretrained weights and change last layer
pretrained = torch.load("/beegfs/yd1282/DLSP20Dataset/pretrained_recons/PreTrained_gen_ckpt.pth")
generator.load_state_dict(pretrained['g_state_dict'])
generator.model[26] = nn.Conv2d(64, 1, kernel_size=(7, 7), stride=(1, 1))
generator = generator.to(device)
discriminator.apply(init_weights)

generator.train()
discriminator.train()

# We combine GAN Loss and L1 Loss to attain Total Loss
criterion_gan = GANLoss('BCE', real_label=real_label, gen_label=gen_label).to(device)
criterion_L1 = nn.L1Loss().to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

####################################
########## TRAINING LOOP ###########
####################################

n_batches = len(trainloader)
for epoch in range(start_epoch, n_epochs):
    iters = 1
    for image, target, road_image in tqdm(trainloader):
        """
        TRAIN DISCRIMINATOR ON BOTH REAL + GENERATED SAMPLES
        """
        real_18ch = torch.stack(image).to(device)
        real_map = []
        for i in range(real_18ch.shape[0]):
            real_map.append(target[i]['masks'])
        real_map = torch.stack(real_map).to(device)

        gen_map = generator(real_18ch)
        
        real_19ch = torch.cat((real_18ch, real_map), 1).to(device)
        gen_19ch = torch.cat((real_18ch, gen_map), 1).to(device)

        output_real = discriminator(real_19ch)
        output_gen = discriminator(gen_19ch.detach())
        
        d_real_mean = output_real.mean().item()
        d_gen_mean_pre = output_gen.mean().item()

        # Make label tensors.
        target_real = torch.Tensor([real_label]).expand_as(output_real).to(device)
        target_gen = torch.Tensor([gen_label]).expand_as(output_gen).to(device)

        # Compute loss.
        loss_d_real = criterion_gan(output_real, target_is_real=True)
        loss_d_gen = criterion_gan(output_gen, target_is_real=False)
        loss_d_total = (loss_d_real + loss_d_gen)

        discriminator.zero_grad()
        loss_d_total.backward()
        optim_D.step()
        
        """
        TRAIN GENERATOR
        """
        output_gen = discriminator(gen_19ch)
        loss_g_gan = criterion_gan(output_gen, target_is_real=True)
        loss_g_L1 = L1_lambda * criterion_L1(gen_map, real_map)
        loss_g_total = loss_g_gan + loss_g_L1
        d_gen_mean_post = output_gen.mean().item()
        
        generator.zero_grad()
        loss_g_total.backward()
        optim_G.step()
        
        """
        COMPILE TRAINING STATISTICS
        """
        losses_d.append(loss_d_total.item())
        losses_g.append(loss_g_total.item())
        
        if iters % update_stats_rate == 1:
            print_train_stats(epoch, n_epochs, iters, n_batches,
                              loss_d_total, loss_g_total, d_real_mean, d_gen_mean_pre, d_gen_mean_post)

        iters += 1
    
    # Add generated image to intermediate images list.
    with torch.no_grad():
        gen_map = generator(real_18ch).detach().cpu()
    intermediate_images.append(vutils.make_grid(gen_map, normalize=True))
    
    save_checkpoint(save_checkpoint_path,
                    {'epoch': epoch + 1,
                     'g_state_dict': generator.state_dict(),
                     'd_state_dict': discriminator.state_dict(),
                     'optim_g' : optim_G.state_dict(),
                     'optim_d' : optim_D.state_dict(),
                     'losses_g' : losses_g,
                     'losses_d' : losses_d,
                     'intermediate_images' : intermediate_images,                     
                    })
    
        
