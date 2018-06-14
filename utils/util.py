from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.cm as cm

# Converts a Tensor into a Numpy array

def tensor2im(image_tensor, imtype=np.uint8):

    image_numpy = image_tensor.data.cpu().numpy()
    if len(list(image_tensor.size())) == 4:
        image_numpy = np.squeeze(image_numpy,0)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.squeeze(image_numpy)

    return image_numpy.astype(imtype)

def tensor2im_(image_tensor, imtype=np.uint8):

    image_tensor = image_tensor.squeeze(0)
    image_numpy = image_tensor.data.cpu().numpy()
    if len(list(image_tensor.size())) == 4:
        image_numpy = np.squeeze(image_numpy,0)

    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) #original
    image_numpy *= (225.0/image_numpy.max())
    image_numpy = np.squeeze(image_numpy)

    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):

    image_numpy = np.squeeze(image_numpy).astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_image_(image_numpy, image_path):
    image_numpy = np.squeeze(image_numpy).astype(np.uint8) #
    h,w = image_numpy.shape

    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_numpy, cmap='plasma')
    plt.savefig(image_path, dpi = 300) 
    plt.close()

def save_image__(image_numpy, image_path):

    image_numpy = np.squeeze(image_numpy).astype(np.uint8) #
    h,w = image_numpy.shape

    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_numpy, cmap='Greys')
    plt.savefig(image_path, dpi = 300) 
    plt.close()

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
