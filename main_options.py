#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import argparse
import os

class MainOptions():
	def __init__(self):
		self.parser =  argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):

		# experiment specifics
		self.parser.add_argument('--name',type=str,default='monodepth',help='name of experiment')
		self.parser.add_argument('--gpu_ids', type=str, default=0, help='gpu ids: e.g. 1. use -1 for CPU')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
		self.parser.add_argument('--use_dropout', action='store_true', default=True, help='use dropout for the generator')

        # input/output sizes
		self.parser.add_argument('--batchsize',type=int,default=4,help='input batch size')
		self.parser.add_argument('--input_height',type=int,default=256,help='scale image to this size')
		self.parser.add_argument('--input_width',type=int,default=512,help='scale image to this size')
		self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
		self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels genetator')
		self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels genetator')

        # for setting inputs
		self.parser.add_argument('--dataroot',type=str,default='../dataset/kitti/')
		self.parser.add_argument('--filename',type=str,default='../dataset/filenames/kitti_train_files_mini.txt')  # kitti_temp_files_test.txt / kitti_train_files_mini.txt / kitti_train_files.txt
		self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
		self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
		self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset.')

		# for generator
		self.parser.add_argument('--netG', type=str, default='mononet', help='select arch for netG [mononet]')

		# for training
		self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
		self.parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
		self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
		self.parser.add_argument('--lr_G', type=float, default=0.0004, help='initial learning rate for adam of G')
		self.parser.add_argument('--headstart', type=int, default= -1, help='# of headstart training iteration for G. # epoch. -1 for off')
		self.parser.add_argument('--headstart_switch', type=int, default= 0, help='switch to turn on discriminator. 0 for off and -1 for on')

		# loss parameters
		self.parser.add_argument('--alpha_image_loss', type=float, default= 0.85, help='weight between SSIM and L1')
		self.parser.add_argument('--disp_grad_loss_wt', type=float, default= 0.1, help='disparity smoothness weight')
		self.parser.add_argument('--lr_loss_wt', type=float, default= 0.0, help='left right consistency weight')

		# load model
		self.parser.add_argument('--load', action='store_true', default=False, help='if true, load model')
		self.parser.add_argument('--ckpt_folder', type=str, default='backup_ckpts', help='name of the checkpoint folder')
		self.parser.add_argument('--which_model', type=str, default='our_gan', help='name of the model')
		self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to be loaded eg. 10,20,30,40,50. if zero, load latest model')

		# train or test
		self.parser.add_argument('--isTrain', action='store_true', default=True, help='if true, training else testing')

		# test files
		self.parser.add_argument('--ext_test', action='store_true', default=False, help='if true, reading external test files')
		self.parser.add_argument('--filename_test',type=str, default='../dataset/filenames/kitti_temp_files_test.txt', help='test filename') #kitti_temp_files_test.txt
		self.parser.add_argument('--ext_filename_test',type=str, default='./checkpoints/monodepth/test_img_in/filename_test.txt', help='ext test filename')
		self.parser.add_argument('--ext_test_in', type=str, default='/media/Data/Alwyn/monodepth/checkpoints/DepthGAN/test_img_in', 
																												help='location of external test files')

		# save model
		self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
		self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        
		# display
		self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
		self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
		self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
		self.parser.add_argument('--use_html', action='store_true', default=True, help='if specified, it will use html page or else tensorboard logging')
		self.parser.add_argument('--individual_loss_disp', action='store_true', default=True, help='if true, will plot individual loss')
		self.parser.add_argument('--save_fake', action='store_true',default=False, help='if true save disp maps')

		# debug
		self.parser.add_argument('--debug', action='store_true',default=False, help='if true, find gradient in computational graph')
		self.parser.add_argument('--graph', action='store_true',default=False, help='if true, build graph in tensorboard')

		# display progress bar
		self.parser.add_argument('--progress', action='store_true',default=False , help='if true, display progress bar')

		self.initialized = True
	
	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt