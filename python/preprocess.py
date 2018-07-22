import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from skimage.transform import resize
from tqdm import tqdm as tqdm

from python.config import masks_path, train_gen_args, test_gen_args
from python.config import n_h, n_v, n_c
from python.config import n_train, n_test
from python.utils import load_depth_file

class ImageLoader:
    def __init__(self, train_path, test_path, depth_file):
        self.train_path = train_path
        self.test_path = test_path
        self.depth_file = depth_file
        self.df_depth = load_depth_file(depth_file)
        self.X_train_img, self.X_train_feat, self.Y_train, self.label_train = self.load_train_image_with_features(self.train_path)
        self.X_test_img, self.X_test_feat, self.label_test = self.load_test_image_with_features(self.test_path)
        self.feat_mean, self.feat_std = self.get_mean_std(self.X_train_feat)
        self.train_gen = ImageDataGenerator(**train_gen_args)
        self.test_gen = ImageDataGenerator(**test_gen_args)
        self.train_gen.fit(self.X_train_img)
        self.test_gen.fit(self.X_test_img)
        self.n_train = self.X_train_img.shape[0]
        self.n_test = self.X_test_img.shape[0]

    def get_mean_std(self, array):
        array_mean = array.mean(axis = 0, keepdims = True)
        array_std = array.std(axis = 0, keepdims = True)
        return array_mean, array_std

    def standardize(self, array, array_mean, array_std):
        array -= array_mean
        array /= array_std
        return array

    def load_train_image(self, train_path):
        X_train = np.zeros((n_train, n_h, n_v, n_c))
        Y_train = np.zeros((n_train, n_h, n_v, n_c))
        label_train = []
        for idx, image_name in tqdm(enumerate(os.listdir(train_path))):
            img = load_img(os.path.join(train_path, image_name), grayscale=True)
            msk = load_img(os.path.join(masks_path, image_name), grayscale=True)
            img_array = img_to_array(img)
            img_array = resize(img_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            msk_array = img_to_array(msk)
            msk_array = resize(msk_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            X_train[idx, :, :, :] = img_array
            Y_train[idx, :, :, :] = msk_array
            label_train.append(image_name.split('.')[0])
        return X_train, Y_train, label_train

    def load_test_image(self, test_path):
        X_test = np.zeros((n_test, n_h, n_v, n_c))
        label_test = []
        for idx, image_name in tqdm(enumerate(os.listdir(test_path))):
            img = load_img(os.path.join(test_path, image_name), grayscale=True)
            img_array = img_to_array(img)
            img_array = resize(img_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            X_test[idx, :, :, :] = img_array
            label_test.append(image_name.split('.')[0])
        return X_test, label_test

    def load_train_image_with_features(self, train_path):
        X_train_img = np.zeros((n_train, n_h, n_v, n_c))
        X_train_feat = np.zeros((n_train, 1), dtype=np.float32)
        Y_train = np.zeros((n_train, n_h, n_v, n_c))
        label_train = []
        for idx, image_path in tqdm(enumerate(os.listdir(train_path))):
            image_name = image_path.split('.')[0]
            img = load_img(os.path.join(train_path, image_path), grayscale=True)
            msk = load_img(os.path.join(masks_path, image_path), grayscale=True)
            img_array = img_to_array(img)
            img_array = resize(img_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            msk_array = img_to_array(msk)
            msk_array = resize(msk_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            X_train_img[idx, :, :, :] = img_array
            X_train_feat[idx, :] = self.df_depth.loc[self.df_depth.id == image_name, 'z'].values[0]
            Y_train[idx, :, :, :] = msk_array
            label_train.append(image_name)
        return X_train_img, X_train_feat, Y_train, label_train

    def load_test_image_with_features(self, test_path):
        X_test_img = np.zeros((n_test, n_h, n_v, n_c))
        X_test_feat = np.zeros((n_test, 1))
        label_test = []
        for idx, image_path in tqdm(enumerate(os.listdir(test_path))):
            image_name = image_path.split('.')[0]
            img = load_img(os.path.join(test_path, image_path), grayscale=True)
            img_array = img_to_array(img)
            img_array = resize(img_array, (n_h, n_v, n_c), mode='constant', preserve_range=True)
            X_test_img[idx, :, :, :] = img_array
            X_test_feat[idx, :] = self.df_depth.loc[self.df_depth.id == image_name, 'z'].values[0]
            label_test.append(image_name)
        return X_test_img, X_test_feat, label_test

    def get_X_train_img(self):
        img = self.X_train_img.copy()
        img = self.train_gen.standardize(img)
        return img

    def get_X_train_feat(self):
        array = self.X_train_feat.copy()
        array = self.standardize(array, self.feat_mean, self.feat_std)
        return array

    def get_Y_train(self):
        img = self.Y_train.copy()
        return img / 255.

    def get_label_train(self):
        return self.label_train

    def get_X_test_img(self):
        img = self.X_test_img.copy()
        img = self.test_gen.standardize(img)
        return img

    def get_X_test_feat(self):
        array = self.X_test_feat.copy()
        array = self.standardize(array, self.feat_mean, self.feat_std)
        return array

    def get_label_test(self):
        return self.label_test

    #TODO might still be buggy. The Y Train is not done properly
    def randomize_train(self, X_train, Y_train):
        random_X_train = np.zeros(X_train.shape)
        random_Y_train = np.zeros(Y_train.shape)
        n_train = X_train.shape[0]
        n_h, n_v = X_train.shape[1], X_train.shape[2]
        for idx in range(n_train):
            random_transform = self.train_gen.get_random_transform((n_h, n_v))
            random_X_train[idx, :, :, :] = self.train_gen.apply_transform(X_train[idx, :, :, :], random_transform)
            random_Y_train[idx, :, :, :] = self.train_gen.apply_transform(Y_train[idx, :, :, :], random_transform)
        return random_X_train, random_Y_train
