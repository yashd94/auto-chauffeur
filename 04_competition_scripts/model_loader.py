"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

from __future__ import absolute_import, division, print_function

from scipy.spatial import distance as dist
import numpy as np
import cv2

import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision

from models import Generator

# import your model class
# import ...
to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
# Put your transform function here, we will use it for our dataloader
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tr, br, tl, bl], dtype="float32")

class ModelLoader():
    # Fill the information for your team
    team_name = 'Crazy Glitch Asians'
    round_number = 3
    team_number = 46
    team_member = ['Alexander Gao', 'Yash Deshpande', 'Annie Wang']
    contact_email = 'yd1282@nyu.edu'

    def __init__(self, model_file_RM = "cgan_generator_L1_PatchGAN_100epoch.pth", 
                 model_file_OM = "ObjMap_GAN_ckpt.pth"):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.model_file_RM = model_file_RM
        self.model_file_OM = model_file_OM
        pass

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        input_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256,256)),
                                              transforms.ToTensor()
                                             ])

        output_transform = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((800,800)),
                                               transforms.ToTensor()
                                              ])
            
        samples = samples.squeeze(0)
        imgs = []
        for i in range(samples.shape[0]):
            imgs.append(samples[i, :, :, :])
        
        x = torch.cat(imgs, dim = 0).unsqueeze(0)
            
        generator = Generator(18, 1, 64, 7)
        generator.load_state_dict((torch.load(self.model_file_OM)['g_state_dict']))
        generator = generator.cuda()
        generator.eval()
        
        # Resize output maps to 800 x 800 and append to a list
        output_maps = generator(x)
        return detect_objects(output_maps)

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]

        input_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256,256)),
                                              transforms.ToTensor()
                                             ])

        output_transform = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((800,800)),
                                               transforms.ToTensor()
                                              ])

        #Transform input to match dimensions required by model.
        batch_images = []
        for i in range(samples.size(0)):
            sample_images = []
            for j in range(samples[i].size(0)):
                sample_images.append(input_transform(samples[i][j].cpu()))
            sample = torch.cat(sample_images)
            batch_images.append(sample)            
        x = torch.stack(batch_images)

        #Instantiate generator and load weights.
        model_weights_path = self.model_file_RM #'weights/cgan_generator_L1_PatchGAN_100epoch.pth'
        generator = Generator(18, 1, 64, 9)
        generator.load_state_dict(torch.load(model_weights_path))
        generator.eval()

        output = generator(x)
        output = output.squeeze(0)

        output_images = []
        for i in range(output.size(0)):
            output_images.append(output_transform(output[i]))
        output = torch.stack(output_images).squeeze()

        return output


# def main():
    """
    Sanity check purposes only. 
    """
    # samples = torch.ones([2, 6, 3, 256, 306])
    # model = ModelLoader()
    # model.get_binary_road_map(samples)

# if __name__ == "__main__":
#     main()
