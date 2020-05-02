# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    """Superclass for NYU Deep Learning Autonomous Driving Dataset
    """
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[882.61644117/4, 0,              621.63358525/4, 0],
                           [0,              882.61644117/4, 621.63358525/4, 0],
                           [0,              0,              1,              0],
                           [0,              0,              0,              1]], dtype=np.float32)

        self.full_res_shape = (320, 256)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color