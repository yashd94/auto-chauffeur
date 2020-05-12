# Written by Team Crazy Glitch Asians, NYU Deep Learning Spring 2020

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def init_weights(m):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def save_checkpoint(fpath, state):
    torch.save(state, fpath)

def print_sys_mem_usage():
    print("Memory usage (RAM):\n")
    !free -h
    print("\nGPU Memory Usage:\n")
    !nvidia-smi --query-gpu=memory.used --format=csv
    !nvidia-smi --query-gpu=memory.total --format=csv
    print("\n")
    
def print_train_stats(epoch, n_epochs, i, n_batches, loss_d_total, loss_g_total, d_real_mean, d_gen_mean_pre, d_gen_mean_post):
    print('Epoch: [%d/%d]\tBatch: [%d/%d]\n' % (epoch + 1, n_epochs, i, n_batches),
          'Loss_D: %.4f\tLoss_G: %.4f\n' % (loss_d_total.item(), loss_g_total.item()),
          'D(real): %.4f  ||  D(gen) Pre-step: %.4f  ||  D(gen) Post-step: %.4f\n' %
          (d_real_mean, d_gen_mean_pre, d_gen_mean_post))

def detect_objects(output_maps):
    
    import cv2

    to_image = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    output_maps_resized = []
    for i in range(output_maps.shape[0]):
        output_maps_resized.append(to_image(output_maps[i].cpu()).resize((800, 800)))
    
    # Detect rectangular bounding boxes
    d_kernel = np.ones((8, 8), np.uint8)
    e_kernel = np.ones((8, 8), np.uint8)
    all_boxes = []
#         print("len op: {}".format(len(output_maps_resized)))
    for i in range(len(output_maps_resized)):

        s = output_maps_resized[i]

        dilated = cv2.dilate(np.array(s), kernel = d_kernel)
        eroded = cv2.erode(dilated, kernel = e_kernel)

        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for j in range(len(contours)):

            rect = cv2.minAreaRect(contours[j])
            box = cv2.boxPoints(rect)
            new_box = order_points(box)
            new_box_ord = (box - 400)/10
            boxes.append(new_box_ord.transpose())

        all_boxes.append(torch.from_numpy(np.array(boxes)))
    
    all_boxes = tuple(all_boxes)
#         print("Pred tuple length: {}".format(len(all_boxes)))
    
    f_shape = all_boxes[0].shape[0]
#         print("1st shape: {}".format())
    if f_shape == 0:
        all_boxes = tuple(torch.rand(1, 2, 2, 4))
#             print("abs: {}".format(all_boxes[0].shape))
        return all_boxes
    else:
#             print("abs: {}".format(all_boxes[0].shape))
        return all_boxes
