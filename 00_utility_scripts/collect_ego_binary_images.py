#Written by Alexander Gao

import os
import sys
from PIL import Image
from helper_extended import *
import torchvision

def convert_and_save_roadmap(img_path, scene_dir, sample_dir, output_path):
	ego_image = Image.open(img_path)
	ego_image = torchvision.transforms.functional.to_tensor(ego_image)

	road_map = convert_map_to_road_map(ego_image)
	road_map = road_map.data.numpy()
	road_map = Image.fromarray(road_map)

	map_filename = scene_dir + '_' + sample_dir + '_' + 'road_map.png'
	map_outpath = os.path.join(output_path, map_filename)
	road_map.save(map_outpath)

def main():
	"""
	This script should be run in the command line and provided the following arguments:
	Argument 1: path to 'data' directory provided as a resource for the DL competition.
	Argument 2: path to output folder where ego maps that have been converted to roadmaps will be collected and saved.
	"""
	data_dir_path = sys.argv[1]
	output_path = sys.argv[2]
	print("\n\nData directory: ", data_dir_path)
	print("Output directory: ", output_path, "\n\n")
	for scene_dir in os.listdir(data_dir_path):
		scene_dir_path = os.path.join(data_dir_path, scene_dir)
		if os.path.isdir(scene_dir_path):
			for sample_dir in os.listdir(scene_dir_path):
				sample_dir_path = os.path.join(scene_dir_path, sample_dir)
				for img in os.listdir(sample_dir_path):
					img_path = os.path.join(sample_dir_path, img)
					if img == 'ego.png':
						convert_and_save_roadmap(img_path, scene_dir, sample_dir, output_path)

if __name__ == "__main__":
	main()
