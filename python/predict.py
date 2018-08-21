import numpy as np
import pandas as pd

from tqdm import tqdm
from python.config import ori_n_c, ori_n_v, ori_n_h, n_test
from keras.preprocessing.image import ImageDataGenerator

def threshold_prediction(Y_pred, threshold):
    return np.round((Y_pred > threshold).astype(np.uint8))

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    img = img.squeeze()
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def TTA_prediction(model, X_pred):
    datagen = ImageDataGenerator()
    assert X_pred.shape == (n_test, ori_n_h, ori_n_v, ori_n_c)
    Y_pred = np.zeros((n_test, ori_n_h, ori_n_v, ori_n_c))
    for idx in tqdm(range(n_test)):
        X_pred_indiv = X_pred[idx, :, :, :]
        Y_pred_indiv = np.zeros((8, ori_n_h, ori_n_v, ori_n_c))
        for theta in [0., 90., 180., 270.]:
            for is_flip_horizontal in [True, False]:
                img = datagen.apply_transform(X_pred_indiv,
                {'theta': theta,
                 'flip_horizontal': is_flip_horizontal})
                mask = model.predict(img, verbose = 1)
                Y_pred_indiv[idx, :, :, :] = datagen.apply_transform(mask,
                {'theta': np.abs(360. - theta),
                 'flip_horizontal': is_flip_horizontal})
        Y_pred[idx, :, :, :] = np.mean(Y_pred_indiv, axis=0)
    print Y_pred

def generate_dict_pred(Y_pred, label_pred, threshold):
    assert Y_pred.shape == (n_test, ori_n_h, ori_n_v, ori_n_c)
    assert len(label_pred) == n_test
    Y_pred_thres = threshold_prediction(Y_pred, threshold)
    dict_pred = {}
    for idx in tqdm(range(n_test)):
        dict_pred[label_pred[idx]] = RLenc(Y_pred_thres[idx, :, :, :])
    return dict_pred

def generate_submission(dict_pred, output_path):
    sub = pd.DataFrame.from_dict(dict_pred, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(output_path)

