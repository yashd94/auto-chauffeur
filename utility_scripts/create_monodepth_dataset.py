#Written by Alexander Gao

import os
import sys
from PIL import Image
from helper import *
import torchvision

def resize_and_crop(color):
	color = color.resize((320, 268), Image.BILINEAR)
	color = color.crop((0, 0, 320, 256))
	return color

def convert_and_save_color(img_path, scene_dir, camera, sample_dir, output_path, trainfiles, valfiles):
	color = Image.open(img_path)
	color = resize_and_crop(color)

	scene_camera_dir = scene_dir + '_' + camera

	color_num = sample_dir.split('_')[-1]
	color_filename = ['0']*10
	for i, char in enumerate(color_num):
		color_filename[i - len(color_num)] = char
	color_filename = "".join(color_filename)

	if int(scene_dir.split('_')[-1]) > 8 and color_num != '0' and color_num != '125':
		trainfiles.write(scene_camera_dir + ' ' + color_filename + '\n')
	elif color_num != '0' and color_num != '125':
	 	valfiles.write(scene_camera_dir + ' ' + color_filename  + '\n')

	color_filename += ".jpg"
	os.makedirs(os.path.join(output_path, scene_camera_dir), exist_ok=True)
	color_outpath = os.path.join(output_path, scene_camera_dir, color_filename)
	color.save(color_outpath)

def main():
	"""
	This script should be run in the command line and provided the following arguments:
	Argument 1: path to 'data' directory provided as a resource for the DL competition.
	Argument 2: path to output folder
	Argument 3: path to 'train_files.txt' a list of training files
	Argument 3: path to 'val_files.txt' a list of validation files
	"""
	data_dir_path = sys.argv[1]
	output_path = sys.argv[2] #e.g. /Users/alexandergao/Documents/DeepLearning/Competition/git/monodepth2/data_nyu
	trainfiles_path = sys.argv[3]
	valfiles_path = sys.argv[4]

	trainfiles = open(trainfiles_path, 'a')
	valfiles = open(valfiles_path, 'a')

	print("\n\nData directory: ", data_dir_path)
	print("Output directory: ", output_path, "\n\n")
	for scene_dir in os.listdir(data_dir_path):
		scene_dir_path = os.path.join(data_dir_path, scene_dir)
		if os.path.isdir(scene_dir_path):
			for sample_dir in os.listdir(scene_dir_path):
				sample_dir_path = os.path.join(scene_dir_path, sample_dir)
				for img in os.listdir(sample_dir_path):
					img_path = os.path.join(sample_dir_path, img)
					print(img_path)
					if img == "CAM_BACK_LEFT.jpeg":
						convert_and_save_color(img_path, scene_dir, 'BACK_LEFT', sample_dir, output_path, trainfiles, valfiles)
					if img == "CAM_BACK_RIGHT.jpeg":
						convert_and_save_color(img_path, scene_dir, 'BACK_RIGHT', sample_dir, output_path, trainfiles, valfiles)
					if img == "CAM_BACK.jpeg":
						convert_and_save_color(img_path, scene_dir, 'BACK', sample_dir, output_path, trainfiles, valfiles)
					if img == "CAM_FRONT_LEFT.jpeg":
						convert_and_save_color(img_path, scene_dir, 'FRONT_LEFT', sample_dir, output_path, trainfiles, valfiles)
					if img == "CAM_FRONT_RIGHT.jpeg":
						convert_and_save_color(img_path, scene_dir, 'FRONT_RIGHT', sample_dir, output_path, trainfiles, valfiles)
					if img == "CAM_FRONT.jpeg":
						convert_and_save_color(img_path, scene_dir, 'FRONT', sample_dir, output_path, trainfiles, valfiles)

	trainfiles.close()
	valfiles.close()

if __name__ == "__main__":
	main()
