#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import time
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from main_options import MainOptions
from depth_modelv2 import model
from utils.visualizer import Visualizer

opt = MainOptions().parse() 

# loading data
if opt.isTrain:
	# loading training images
	from dataloader import StereoDataloader
	data_loader = StereoDataloader(opt) # create dataloader 
	train_data = DataLoader(data_loader, batch_size=opt.batchsize, shuffle=True)#, num_workers=1)
	dataset_size = len(data_loader)
	print('#training images: %d' %dataset_size)
else:
	# loading test images
	from dataloader_test import StereoDataloader_test
	data_loader = StereoDataloader_test(opt) # create dataloader 
	test_data = DataLoader(data_loader, batch_size=1, shuffle=False)#, num_workers=1)
	dataset_size = len(data_loader)
	print('#test images: %d' %dataset_size)

if opt.isTrain:
	start_epoch, epoch_iter = 1, 0
	total_steps = (start_epoch - 1) * dataset_size + epoch_iter
	iter_ = 1
else:
	opt.load = True

# create/load model
model = model(opt)
visualizer = Visualizer(opt)

if opt.isTrain:
	print('\nTraining started...')
	# for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
	for epoch in range(start_epoch, opt.niter+1):
		# epoch start time
		epoch_start_time = time.time()
		# opt.current = epoch

		for i, data in enumerate(train_data, start=epoch_iter):
			iter_start_time = time.time()
			total_steps += opt.batchsize
			epoch_iter += opt.batchsize

			# whether to collect output images
			opt.save_fake = total_steps % opt.display_freq == 0

			if opt.save_fake:
				print('\nEpoch: %d, Iteration: %d' %(epoch, total_steps))

			# when to start dis traning
			if epoch > opt.headstart: 
					opt.headstart_switch = -1

			# forward
			loss_G = model(Variable(data['left_img']), Variable(data['right_img']))

			# backward G
			model.optimizer_G.zero_grad()
			loss_G.backward()
			model.optimizer_G.step()

			# # display input & output and save ouput images
			if opt.save_fake:
				result_img = model.get_result_img(Variable(data['left_img']), Variable(data['right_img']))
				visualizer.display_current_results(result_img, epoch, total_steps)

		# epoch end time
		iter_end_time = time.time()
		print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.niter, time.time() - epoch_start_time))

	    # save mdodel
		print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
		model.save(epoch)

else:
	print('Testing started...')
	if opt.progress:
		for i, data in enumerate(tqdm(test_data)):
			gen_disp = model.inference(Variable(data['test_img']))
			result_img = model.get_test_result(gen_disp)
			visualizer.display_test_results(i, result_img)
	else:
		for i, data in enumerate(test_data):
			gen_disp = model.inference(Variable(data['test_img']))
			result_img = model.get_test_result(gen_disp)
			visualizer.display_test_results(i, result_img)
	print('Testing finsihed.')
