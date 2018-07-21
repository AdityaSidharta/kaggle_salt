import pandas as pd
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

def load_depth_file(depth_file):
    df_depth = pd.read_csv(depth_file)
    return df_depth

def resize_img(img, h, v, c):
    n_img = img.shape[0]
    img_resize = np.zeros((n_img, h, v, c))
    for idx in tqdm(range(n_img)):
        img_resize[idx, :, :, :] = resize(img[idx, :, :, :], (h, v, c),mode='constant', preserve_range=True)
    return img_resize
