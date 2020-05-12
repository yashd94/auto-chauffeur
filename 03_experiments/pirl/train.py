"""
Code modified code by Alexander Gao, based on an excellent
open-source implementation by @akwasigroch:
https://github.com/akwasigroch/Pretext-Invariant-Representations

Also successfully managed to contribute to his open source repo by profiling his
code and detecting a significant performance bottleneck when generating random indices
for negative samples.  Training cycles improved by an order of magnitude.
"""

import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from torchvision.datasets import DatasetFolder

from utils import AverageMeter, Progbar, Memory, ModelCheckpoint, NoiseContrastiveEstimator, pil_loader, Logger
from dataset import RotationDataset
from encoder import Resnet18_Encoder

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = ("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/home/cassava_media_arts/data'
checkpoint_dir = '/home/cassava_media_arts/tmp/pirl_rotation_models'
log_filename = 'pretraining_log_rotation'

resume_from_previous_state = True
starting_epoch = 16
previous_state_path = '/home/cassava_media_arts/tmp/pirl_rotation_models/epoch_' + str(starting_epoch)

lr = 0.001
negative_nb = 8000 # number of negative examples in NCE
unlabeled_scene_index = np.arange(130)

dataset = RotationDataset(data_dir, unlabeled_scene_index)
train_loader = DataLoader(dataset, shuffle = True, batch_size = 180, num_workers=2, pin_memory=True)

encoder = Resnet18_Encoder().to(device)
optimizer = optim.SGD(encoder.parameters(), lr=lr, momentum=0.9)

if resume_from_previous_state:
    previous = torch.load(previous_state_path)
    encoder.load_state_dict(previous['model'])
    optimizer.load_state_dict(previous['optimizer'])

memory = Memory(size=len(dataset), weight=0.5, device=device)
memory.initialize(encoder, train_loader)

checkpoint = ModelCheckpoint(mode='min', directory=checkpoint_dir)
noise_contrastive_estimator = NoiseContrastiveEstimator(device)
logger = Logger(log_filename)

loss_weight = 0.5


###########################
######## TRAINING #########
###########################

for epoch in range(starting_epoch + 1, 1000):
    print('\nEpoch: {}'.format(epoch))
    memory.update_weighted_count()
    train_loss = AverageMeter('train_loss')
    bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])
    
    for step, batch in enumerate(train_loader): 


        # prepare batch
        images = batch['original'].to(device)
        rotation = batch['rotation'].to(device)
        index = batch['index']
        representations = memory.return_representations(index).to(device).detach()
        # zero grad
        optimizer.zero_grad()
        
        #forward, loss, backward, step
        output = encoder(images = images, rotation = rotation, mode = 1)
                
        loss_1 = noise_contrastive_estimator(representations, output[1], index, memory, negative_nb = negative_nb)
        loss_2 = noise_contrastive_estimator(representations, output[0], index, memory, negative_nb = negative_nb) 
        loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2
        
        loss.backward()
        optimizer.step()

        #update representation memory
        memory.update(index, output[0].detach())
        
        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        bar.update(step, values=[('train_loss', train_loss.return_avg())])
        
        #save model if improved
        checkpoint.save_model(encoder, optimizer, train_loss.return_avg(), epoch)
    
    logger.update(epoch, train_loss.return_avg())


