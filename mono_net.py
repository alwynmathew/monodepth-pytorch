#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class mono_net(nn.Module): 	# vgg version
    def __init__(self, input_nc, output_nc):
        super(mono_net, self).__init__()

        self.output_nc = output_nc

        self.downconv_1 = self.conv_down_block(input_nc,32,7)
        self.downconv_2 = self.conv_down_block(32,64,5)
        self.downconv_3 = self.conv_down_block(64,128,3)
        self.downconv_4 = self.conv_down_block(128,256,3)
        self.downconv_5 = self.conv_down_block(256,512,3)
        self.downconv_6 = self.conv_down_block(512,512,3)
        self.downconv_7 = self.conv_down_block(512,512,3)

        self.upconv_7 = self.conv_up_block(512,512)
        self.upconv_6 = self.conv_up_block(512,512)
        self.upconv_5 = self.conv_up_block(512,256)
        self.upconv_4 = self.conv_up_block(256,128)
        self.upconv_3 = self.conv_up_block(128,64)
        self.upconv_2 = self.conv_up_block(64,32)
        self.upconv_1 = self.conv_up_block(32,16)   

        self.conv_7 = self.conv_block(1024,512)
        self.conv_6 = self.conv_block(1024,512)
        self.conv_5 = self.conv_block(512,256)
        self.conv_4 = self.conv_block(256,128)
        self.conv_3 = self.conv_block(130,64)
        self.conv_2 = self.conv_block(66,32)
        self.conv_1 = self.conv_block(18,16)

        self.get_disp4 = self.disp_block(128)
        self.get_disp3 = self.disp_block(64)
        self.get_disp2 = self.disp_block(32)
        self.get_disp1 = self.disp_block(16)

    def conv_down_block(self, in_dim, out_dim, kernal):

        conv_down_block = []
        conv_down_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernal, stride=1, padding=int((kernal-1)/2)),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w
        conv_down_block += [nn.Conv2d(out_dim, out_dim, kernel_size=kernal, stride=2, padding=int((kernal-1)/2)), nn.BatchNorm2d(out_dim), nn.ELU()]   # h,w -> h/2,w/2

        return nn.Sequential(*conv_down_block)

    def conv_up_block(self, in_dim, out_dim):

        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def conv_block(self, in_dim, out_dim):

        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),nn.ELU()]      # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def disp_block(self, in_dim):

        disp_block = []
        disp_block += [nn.Conv2d(in_dim, self.output_nc, kernel_size=3, stride=1, padding=1),nn.Sigmoid()]  # h,w -> h,w

        return nn.Sequential(*disp_block)

    def upsample_(self, disp, ratio):

        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def forward(self, x): 
    							# 3x256x512
    	conv_1 = self.downconv_1(x) # 32x128x256
    	conv_2 = self.downconv_2(conv_1) # 64x64x128
    	conv_3 = self.downconv_3(conv_2) # 128x32x64
    	conv_4 = self.downconv_4(conv_3) # 256x16x32
    	conv_5 = self.downconv_5(conv_4) # 512x8x16
    	conv_6 = self.downconv_6(conv_5) # 512x4x8
    	conv_7 = self.downconv_7(conv_6) # 512x2x4

    	conv7_up = self.upsample_(conv_7, 2) # 512x4x8
    	upconv_7 = self.upconv_7(conv7_up) # 512x4x8
    	concat_7 = torch.cat([upconv_7,conv_6], 1) # 1024x4x8
    	iconv_7 = self.conv_7(concat_7) # 512x4x8

    	iconv7_up = self.upsample_(iconv_7, 2) # 512x8x16
    	upconv_6 = self.upconv_6(iconv7_up) # 512x8x16
    	concat_6 = torch.cat([upconv_6,conv_5], 1) # 1024x8x16
    	iconv_6 = self.conv_6(concat_6) # 512x8x16

    	iconv6_up = self.upsample_(iconv_6, 2) # 512x16x32
    	upconv_5 = self.upconv_5(iconv6_up) # 256x16x32
    	concat_5 = torch.cat([upconv_5,conv_4], 1) # 512x16x32
    	iconv_5 = self.conv_5(concat_5) # 256x16x32

    	iconv5_up = self.upsample_(iconv_5, 2) # 256x32x64
    	upconv_4 = self.upconv_4(iconv5_up) # 128x32x64
    	concat_4 = torch.cat([upconv_4,conv_3], 1) # 256x32x64
    	iconv_4 = self.conv_4(concat_4) # 128x32x64
    	self.disp4 = 0.3 * self.get_disp4(iconv_4) # 2x32x64
    	udisp4 = self.upsample_(self.disp4, 2) # 2x64x128

    	iconv4_up = self.upsample_(iconv_4, 2) # 128x64x128
    	upconv_3 = self.upconv_3(iconv4_up) # 64x64x128
    	concat_3 = torch.cat([upconv_3,conv_2,udisp4], 1) # 130x64x128
    	iconv_3 = self.conv_3(concat_3) # 64x64x128
    	self.disp3 = 0.3 * self.get_disp3(iconv_3) # 2x64x128
    	udisp3 = self.upsample_(self.disp3, 2) # 2x128x256

    	iconv3_up = self.upsample_(iconv_3, 2) # 64x128x256
    	upconv_2 = self.upconv_2(iconv3_up) # 32x128x256
    	concat_2 = torch.cat([upconv_2,conv_1,udisp3], 1) # 66x128x256
    	iconv_2 = self.conv_2(concat_2) # 32x128x256
    	self.disp2 = 0.3 * self.get_disp2(iconv_2) # 2x128x256
    	udisp2 = self.upsample_(self.disp2, 2) # 2x256x512

    	iconv2_up = self.upsample_(iconv_2, 2) # 32x256x512
    	upconv_1 = self.upconv_1(iconv2_up) # 16x256x512
    	concat_1 = torch.cat([upconv_1, udisp2], 1) # 18x256x512
    	iconv_1 = self.conv_1(concat_1) # 16x256x512
    	self.disp1 = 0.3 * self.get_disp1(iconv_1) # 2x256x512

    	return [self.disp1, self.disp2, self.disp3, self.disp4]
