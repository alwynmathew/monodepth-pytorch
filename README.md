# Monodepth in Pytorch

*NOTE: Repo will be clean up and refactored soon. Roll back to initially stage ec97056, if necessary*

Pytorch implementation of unsupervised single image depth prediction using a convolutional neural network. This is ported from the [original tensorflow implementation](https://github.com/mrharicot/monodepth) by [Clément Godard](https://github.com/mrharicot).

<p align="center">
  <img src="http://visual.cs.ucl.ac.uk/pubs/monoDepth/monodepth_teaser.gif" alt="monodepth">
</p>

**Unsupervised Monocular Depth Estimation with Left-Right Consistency**  
[Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
CVPR 2017

For more details:  
[project page](http://visual.cs.ucl.ac.uk/pubs/monoDepth/)  
[arXiv](https://arxiv.org/abs/1609.03677)

## Requirements
This code was tested with Pytorch 0.3, CUDA 8.0 and Ubuntu 16.04.   
For now, you can train only on single GPU.

## Data
This model requires rectified stereo pairs for training. There are two main datasets available:   
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
We used two different split of the data, **kitti** and **eigen**, amounting for respectively 29000 and 22600 training samples, you can find them in the [filenames](utils/filenames) folder.  
You can download the entire raw dataset by running:
```shell
wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/
```
**Warning:** it weights about **175GB**, make sure you have enough space to unzip too!  
To save space you can convert the png images to jpeg.
```shell
find ~/my/output/folder/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
```

### [Cityscapes](https://www.cityscapes-dataset.com)
You will need to register in order to download the data, which already has a train/val/test set with 22973 training images.  
We used `leftImg8bit_trainvaltest.zip`, `rightImg8bit_trainvaltest.zip`, `leftImg8bit_trainextra.zip` and `rightImg8bit_trainextra.zip` which weights **110GB**.

## Training

Set following flags

`--isTrain` to `True `   
`--name` to any name of your choose eg. `monodepth `   
`--netG` to name of the model eg. `monodepth  `  
`--gpu_ids` to GPU id eg. `0`    
`--batchsize` to batchsize eg. `8 `   
`--filename` to location of the text of kitti image location eg. `/dataset/filenames/kitti_train_files.txt  `  

Command to train model:

```
python train.py --name monodepth --netG monodepth --isTrain --gpu_ids 0 --batchsize 8 
--filename ../dataset/filenames/kitti_train_files.txt
```
 Check out other flags at [main_options.py](https://github.com/alwynmathew/monodepth-pytorch/blob/master/main_options.py)  
 
Saved models and ouput disparities after each `display_freq` can be found at `/checkpoints_dir/name/` and `/checkpoints_dir/name/web/images/` respectively.

## Testing  

Set following flags

`--ext_test_in` flag to dir of test images eg. `/monodepth/checkpoints/monodepth/test_img_in `   
`--load` to `True`    
`--ckpt_folder` to path of your saved model eg. `ckpt_backup `   
`--which_model` to model name eg. `monodepth `   
`--which_epoch` to epoch number eg . `30  `  

Command to test model:
```
python train.py --name monodepth --netG monodepth  --gpu_ids 0 
--ext_test_in /monodepth/ckpt_backup/monodepth/test_img_in --load --ckpt_folder ckpt_backup 
--which_model monodepth --which_epoch 30
```
Output of the test can be found at `checkpoints_dir/name/test_img_out` where `checkpoints_dir` and `name` are flags.  Post-processing not yet implemented in the current version.

### Bilinear sampler

Code for ported and in-built bilinear sampler can be found [here](https://github.com/alwynmathew/bilinear-sampler-pytorch).

