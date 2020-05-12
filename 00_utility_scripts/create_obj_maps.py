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

from data_helper_extended import UnlabeledDataset, LabeledDataset
from helper_extended import collate_fn, draw_filled_box

image_folder = sys.argv[1]
annotation_csv = os.path.join(image_folder, 'annotation.csv')

labeled_scene_index = np.arange(106, 134)

labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=torchvision.transforms.ToTensor(),
                                  extra_info=True
                                 )

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)

for sample, target, road_image, extra, scene_id, sample_id in trainloader:
	print(scene_id[0], sample_id[0])
	fig, ax = plt.subplots()
	ax.imshow(np.zeros((800, 800,3)))
	ax.set_axis_off()

	for i, bb in enumerate(target[0]['bounding_box']):
	    draw_filled_box(ax, bb, color='white')

	fig.savefig(image_folder + '/scene_' + str(scene_id[0]) + '/sample_' + str(sample_id[0]) + '/obj_map.png', bbox_inches='tight', pad_inches = 0, dpi = 265)
	plt.close()
