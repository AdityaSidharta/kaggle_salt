import pandas as pd
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from fastkeras.callbacks import LRFinder


def load_depth_file(depth_file):
    df_depth = pd.read_csv(depth_file)
    return df_depth

def resize_img(img, h, v, c):
    n_img = img.shape[0]
    img_resize = np.zeros((n_img, h, v, c))
    for idx in tqdm(range(n_img)):
        img_resize[idx, :, :, :] = resize(img[idx, :, :, :], (h, v, c),mode='constant', preserve_range=True)
    return img_resize

def find_lr(model, X, Y, min_lr=1e-5, max_lr=1e-1, epoch_size=5, batch_size=32):
    lr_finder = LRFinder(min_lr=min_lr,
                         max_lr=max_lr,
                         steps_per_epoch=np.ceil(epoch_size / batch_size),
                         epochs=epoch_size)
    model.fit(X, Y, callbacks=[lr_finder], epochs = epoch_size, batch_size = batch_size)
    lr_finder.plot_loss()