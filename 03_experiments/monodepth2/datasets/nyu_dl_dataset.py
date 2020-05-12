# Written by Team Crazy Glitch Asians, NYU Deep Learning Spring 2020

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
        imscale = 1224 / 320
        self.K = {'FRONT_LEFT' : np.array([[879.03824732/imscale, 0,              		613.17597314/imscale, 0],
        		[0,              			  879.03824732/imscale, 524.14407205/imscale, 0],
        							  [0,              			  0,              		1,              	  0],
        							  [0,              			  0,              		0,              	  1]], dtype=np.float32),
        		  'FRONT' : np.array([[882.61644117/imscale, 	  0,              		621.63358525/imscale, 0],
        							  [0,              			  882.61644117/imscale, 524.38397862/imscale, 0],
        							  [0,              			  0,              		1,              	  0],
        							  [0,             			  0,        	        0,         		      1]], dtype=np.float32),
        		  'FRONT_RIGHT' : np.array([[880.41134027/imscale, 0,              		 618.9494972/imscale, 0],
        							  [0,              			   880.41134027/imscale, 521.38918482/imscale, 0],
        							  [0,              			   0,              		 1,              	   0],
        							  [0,              			   0,              		 0,              	   1]], dtype=np.float32),
        		  'BACK_LEFT' : np.array([[881.28264688/imscale,   0,              		 612.29732111/imscale, 0],
        							  [0,              			   881.28264688/imscale, 521.77447199/imscale, 0],
        							  [0,              			   0,              		 1,              	   0],
        							  [0,              			   0,              		 0,              	   1]], dtype=np.float32),
        		  'BACK' : np.array([[882.93018422/imscale, 	   0,              		 616.45479905/imscale, 0],
        							  [0,              			   882.93018422/imscale, 528.27123027/imscale, 0],
        							  [0,              			   0,              		 1,              	   0],
        							  [0,              			   0,              		 0,              	   1]], dtype=np.float32),
        		  'BACK_RIGHT' : np.array([[881.63835671/imscale, 0,              		607.66308183/imscale, 0],
        							  [0,              			  881.63835671/imscale, 525.6185326/imscale, 0],
        							  [0,              			  0,              		1,              0],
        							  [0,              			  0,              		0,              1]], dtype=np.float32)
        		}
        self.full_res_shape = (320, 256)

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
