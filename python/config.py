import numpy as np
import os

ori_n_h = 101
ori_n_v = 101
ori_n_c = 1

n_h = 128
n_v = 128
n_c = 1

batch_size = 32
n_epoch = 35

min_lr= 1e-5
max_lr= 1e-3
steps_per_epoch= np.ceil(n_epoch/batch_size)
lr_decay=0.9
cycle_length=2
mult_factor=2

smooth = 1.

parent_train_path = '/home/adityasidharta/git/kaggle_salt/train_images/'
parent_masks_path = '/home/adityasidharta/git/kaggle_salt/train_masks/'
parent_test_path = '/home/adityasidharta/git/kaggle_salt/test_images/'
train_path = '/home/adityasidharta/git/kaggle_salt/train_images/train/'
masks_path = '/home/adityasidharta/git/kaggle_salt/train_masks/masks'
test_path = '/home/adityasidharta/git/kaggle_salt/test_images/test'
depth_file = '/home/adityasidharta/git/kaggle_salt/data/depths.csv'
train_file = '/home/adityasidharta/git/kaggle_salt/data/train.csv'

n_train = len(os.listdir(train_path))
n_test = len(os.listdir(test_path))
n_masks = len(os.listdir(masks_path))

train_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True)
test_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True)