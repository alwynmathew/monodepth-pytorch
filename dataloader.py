#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import random
import torchvision.transforms as transforms
from base_dataloader import BaseDataset, get_params, get_transform, normalize

class StereoDataloader(Dataset):

	__left = []
	__right = []

	def __init__(self,opt):
		self.opt = opt
		filename = self.opt.filename
		dataroot = self.opt.dataroot
		arrlenth = 66 + len(dataroot)
		arrlen = '|S'+str(arrlenth)
		arr = np.genfromtxt(filename, dtype=arrlen, delimiter=' ')
		#transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
		#transform_list = [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
		#self.transforms = transforms.Compose(transform_list)
		n_line = open(filename).read().count('\n')
		for line in range(n_line):
			self.__left.append(dataroot + arr[line][0])
			self.__right.append(dataroot + arr[line][1])

	def __getitem__(self, index):
      	
		img1 = Image.open(self.__left[index])
		params = get_params(self.opt, img1.size)

		img1 = Image.open(self.__left[index]).convert('RGB')
		img2 = Image.open(self.__right[index]).convert('RGB')

		arg = random.random() > 0.5
		if arg:
			img1, img2 = self.augument_image_pair(img1, img2)

		transform = get_transform(self.opt, params)      
		img1 = transform(img1)
		img2 = transform(img2)

		input_dict = {'left_img':img1.cuda(), 'right_img':img2.cuda()}

		return input_dict

	def augument_image_pair(self, left_image, right_image):

		left_image = np.asarray(left_image)
		right_image = np.asarray(right_image)
		# print(np.amin(left_image))

		# randomly gamma shift
		random_gamma = random.uniform(0.8, 1.2)
		left_image_aug  = left_image  ** random_gamma
		right_image_aug = right_image ** random_gamma

        # randomly shift brightness
		random_brightness = random.uniform(0.5, 2.0)
		left_image_aug  =  left_image_aug * random_brightness
		right_image_aug = right_image_aug * random_brightness

        # randomly shift color
		# random_colors = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]
		# white = np.ones((left_image.shape[0],left_image.shape[1]))
		# color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
		# left_image_aug  *= color_image
		# right_image_aug *= color_image

        # saturate
		# left_image_aug  = np.clip(left_image_aug,  0, 1)
		# right_image_aug = np.clip(right_image_aug, 0, 1)

		left_image_aug = Image.fromarray(np.uint8(left_image_aug))
		right_image_aug  = Image.fromarray(np.uint8(right_image_aug))

		return left_image_aug, right_image_aug

	def __len__(self):
		return len(self.__left)
