#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F

import mono_net
# from bilinear_torch import * # ported from Godard's code
from bilinear_sampler import * # from Po-Hsun Su
import pytorch_ssim
import utils.util as util
from utils.visualizer import Visualizer

def print_network(net):
	if isinstance(net, list):
		net = net[0]
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class model(nn.Module):
	def __init__(self,opt):
		super(model, self).__init__()
		self.visualizer = Visualizer(opt)
		self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
		self.opt = opt
		netG_input_nc = opt.input_nc #if opt.input_nc != 0 else 3
		self.gpu_ids = opt.gpu_ids

		if self.opt.netG == 'mononet':
			# mononet network
			print('Building mononet network...')
			self.G =  mono_net.mono_net(netG_input_nc, opt.output_nc)
		else:
			raise NotImplementedError('Network %s is not found' % self.opt.netG)
		
		print(self.G)

		if opt.gpu_ids > -1:
			assert(torch.cuda.is_available())   
			self.G.cuda(opt.gpu_ids)
		self.G.apply(weights_init)

		# optimizer G
		params_G = list(self.G.parameters())
		self.optimizer_G = torch.optim.Adam(params_G, lr=opt.lr_G, betas=(opt.beta1, 0.999))

		# init old_lr
		self.old_lr_G=opt.lr_G

		# load from checkpoint
		if opt.load:
			if opt.which_epoch == 0:
				save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.ckpt_folder, opt.which_model, 'latest_net_G.pth')
			else:
				save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.ckpt_folder, opt.which_model, '%d_net_G.pth' % (opt.which_epoch))
			print('Loading model from %s' % save_path)
			if not os.path.isfile(save_path):
				print('%s not exists yet!' % save_path)
			else:
				try:
					self.G.load_state_dict(torch.load(save_path))
				except:   
					pretrained_dict = torch.load(save_path)                
					model_dict = self.G.state_dict()
					try:
						pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
						self.G.load_state_dict(pretrained_dict)
						print('Pretrained network has excessive layers; Only loading layers that are used')
					except:
						print('Pretrained network has fewer layers; The following are not initialized:')
						if sys.version_info >= (3,0):
							not_initialized = set()
						else:
							from sets import Set
							not_initialized = Set()
						for k, v in pretrained_dict.items():                      
							if v.size() == model_dict[k].size():
								model_dict[k] = v

						for k, v in model_dict.items():
							if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
								not_initialized.add(k.split('.')[0])      
						print(sorted(not_initialized))
						self.G.load_state_dict(model_dict)


	def generate_image_left_(self, img, disp):

		# return bilinear_sampler_1d_h(img, -disp)
		return image_warp(img, -disp)

	def generate_image_right_(self, img, disp):

		# return bilinear_sampler_1d_h(img, disp)
		return image_warp(img, -disp)

	def gradient_x(self, img):
		gx = img[:,:,:,:-1] - img[:,:,:,1:]
		return gx

	def gradient_y(self, img):
		gy = img[:,:,:-1,:] - img[:,:,1:,:]
		return gy

	def get_disparity_smoothness(self, disp, input_img):
		disp_gradients_x = [self.gradient_x(d) for d in disp]
		disp_gradients_y = [self.gradient_y(d) for d in disp]

		image_gradients_x = [self.gradient_x(img) for img in input_img]
		image_gradients_y = [self.gradient_y(img) for img in input_img]

		weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
		weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

		smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
		smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

		smoothness_x = [torch.nn.functional.pad(k,(0,1,0,0,0,0,0,0),mode='constant') for k in smoothness_x]
		smoothness_y = [torch.nn.functional.pad(k,(0,0,0,1,0,0,0,0),mode='constant') for k in smoothness_y]

		return smoothness_x + smoothness_y

	# ssim from Godard's code
	def SSIM(self, x, y):
		C1 = 0.01 ** 2
		C2 = 0.03 ** 2

		mu_x = nn.functional.avg_pool2d(x, 3, 1, padding = 0)
		mu_y = nn.functional.avg_pool2d(y, 3, 1, padding = 0)

		sigma_x  = nn.functional.avg_pool2d(x ** 2, 3, 1, padding = 0) - mu_x ** 2
		sigma_y  = nn.functional.avg_pool2d(y ** 2, 3, 1, padding = 0) - mu_y ** 2

		sigma_xy = nn.functional.avg_pool2d(x * y , 3, 1, padding = 0) - mu_x * mu_y

		SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
		SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

		SSIM = SSIM_n / SSIM_d

		return torch.clamp((1 - SSIM) / 2, 0, 1)

	# ssim from Po-Hsun Su
	def SSIM_(self, x, y):

		ssim_loss = pytorch_ssim.SSIM()

		return torch.clamp(1 - ssim_loss(x, y) / 2, 0, 1)

	def scale_pyramid_(self, img, num_scales):
		img = torch.mean(img, 1)
		img = torch.unsqueeze(img, 1)
		scaled_imgs = [img]
		s = img.size()
		h = int(s[2])
		w = int(s[3])
		for i in range(num_scales):
			ratio = 2 ** (i + 1)
			nh = h // ratio
			nw = w // ratio
			temp = nn.functional.upsample(img, [nh, nw], mode='nearest')
			scaled_imgs.append(temp)
		return scaled_imgs

	def scale_pyramid(self, img, num_scales):
		scaled_imgs = [img]
		s = img.size()
		h = int(s[2])
		w = int(s[3])
		for i in range(num_scales - 1):
			ratio = 2 ** (i + 1)
			nh = h // ratio
			nw = w // ratio
			temp = nn.functional.upsample(img, [nh, nw], mode='bilinear')
			scaled_imgs.append(temp)
		return scaled_imgs

	def inference(self, test_input):

		self.test_disp = self.G(test_input)

		return self.test_disp[0]

	def forward(self, input_left, input_right):

		self.left_pyramid = self.scale_pyramid_(input_left, 4)
		self.right_pyramid = self.scale_pyramid_(input_right, 4)
		self.input = input_left

		self.disp_est = self.G(self.input)
		self.disp_left_est = [torch.unsqueeze(d[:,0,:,:], 1) for d in self.disp_est]
		self.disp_right_est = [torch.unsqueeze(d[:,1,:,:], 1) for d in self.disp_est]

		if not self.opt.isTrain:
			return

		## NETWORK OUTPUT
		# GENERATE IMAGES
		self.left_est = [self.generate_image_left_(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)] 
		self.right_est = [self.generate_image_right_(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

		# LR CONSISTENCY
		self.right_to_left_disp = [self.generate_image_left_(self.disp_right_est[i], self.disp_left_est[i]) for i in range(4)]
		self.left_to_right_disp = [self.generate_image_right_(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

		# DISPARITY SMOOTHNESS
		self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid) # get_disparity_smoothness
		self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid) # get_disparity_smoothness

		## BUILD LOSSES
		# IMAGE RECONSTRUCTION
		# L1
		self.l1_left = [torch.abs(self.left_pyramid[i] - self.left_est[i]) for i in range(4)]
		self.l1_recomstruction_loss_left = [torch.mean(l) for l in self.l1_left]
		self.l1_right = [torch.abs(self.right_pyramid[i] - self.right_est[i]) for i in range(4)]
		self.l1_recomstruction_loss_right = [torch.mean(l) for l in self.l1_right]

		self.ssim_loss_left = [self.SSIM_(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
		self.ssim_loss_right = [self.SSIM_(self.right_est[i], self.right_pyramid[i]) for i in range(4)]

		# WEIGTHED SUM
		self.image_loss_right = [0.85 * self.ssim_loss_right[i] + 0.15 * self.l1_recomstruction_loss_right[i] for i in range(4)]
		self.image_loss_left = [0.85 * self.ssim_loss_left[i] + 0.15 * self.l1_recomstruction_loss_left[i] for i in range(4)]
		self.image_loss1 = [(self.image_loss_left[i] + self.image_loss_right[i]) for i in range(4)]
		self.image_loss = sum(self.image_loss1)

		# DISPARITY SMOOTHNESS
		self.disp_left_loss = [torch.mean(torch.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
		self.disp_right_loss = [torch.mean(torch.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
		self.disp_gradient_loss = sum(self.disp_left_loss + self.disp_right_loss)

		# LR CONSISTENCY
		self.lr_left_loss = [torch.mean(torch.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in range(4)]
		self.lr_right_loss = [torch.mean(torch.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
		self.lr_loss = sum(self.lr_left_loss + self.lr_right_loss)

		self.total_loss = ( 
				self.image_loss + 
				self.opt.disp_grad_loss_wt * self.disp_gradient_loss +
				elf.opt.lr_loss_wt * self.lr_loss
				)

		if self.opt.save_fake:
			print('image_loss: %f, disp_loss: %f, lr_loss: %f' % (self.image_loss, self.disp_gradient_loss,
				self.lr_loss))
			print('total loss: %f' % self.total_loss)

		
		self.loss_G = self.total_loss
		if self.opt.save_fake:
			print('G_loss: %f' % self.loss_G)
		
		return self.loss_G

	def get_current_loss(self):
		if self.opt.headstart_switch == -1:
			return OrderedDict([('loss_G', self.loss_G.data.cpu().numpy()), ('loss_D', self.loss_D.data.cpu().numpy())])
		else:
			return OrderedDict([('loss_G', self.loss_G.data.cpu().numpy())])

	def get_result_img(self, input_left, input_right):
	
		input_left_ = util.tensor2im(input_left[1,:,:,:])
		input_right_ = util.tensor2im(input_right[1,:,:,:])
		left_est_im = util.tensor2im(self.left_est[0][1,:,:,:]) 
		right_est_im = util.tensor2im(self.right_est[0][1,:,:,:])
		left_disp = util.tensor2im_(self.disp_left_est[0][1,:,:,:].unsqueeze(0)) 
		right_disp = util.tensor2im_(self.disp_right_est[0][1,:,:,:].unsqueeze(0))

	
		return OrderedDict([
		('input_left', input_left_),
		('input_right', input_right_),
		('left_est', left_est_im), 
		('right_est', right_est_im),
		('left_disp', left_disp),
		('right_disp', right_disp)
		])

	def get_test_result(self, test_output):
		self.test_output=test_output
		test_output = util.tensor2im_(self.test_output[0,:,:,:])
		return OrderedDict([('test_output', test_output)])

	def save(self, which_epoch):
		self.save_network(self.G, 'G', which_epoch, self.gpu_ids)

	def save_network(self, network, network_label, epoch_label, gpu_ids):
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.save_dir, save_filename)
		torch.save(network.cpu().state_dict(), save_path)
		print('Model saved at %s' % save_path)
		if self.opt.gpu_ids > -1 and torch.cuda.is_available():
			network.cuda()

	def update_learning_rate(self):
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = self.opt.lr_G
		print('G: update learning rate: %f -> %f' % (self.old_lr_G, self.opt.lr_G))
		self.old_lr_G = self.opt.lr_G
