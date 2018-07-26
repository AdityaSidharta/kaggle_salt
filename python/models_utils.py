from keras import layers
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Dense

def conv_block(inp, filter_size, is_maxpool = True):
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(inp)
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(c)
    if is_maxpool:
        p = MaxPooling2D((2, 2))(c)
        return c, p
    else:
        return c

def trans_conv_block(inp, c_inp, filter_size):
    u = Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(inp)
    u = concatenate([u, c_inp])
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(u)
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(c)
    return c

def dense_block_bn_dp(x, dense_list, dropout = 0.0):
    dense_list = [dense_list] if type(dense_list) != list else dense_list
    for dense_size in dense_list:
        x = Dense(dense_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    return x

def conv_block_bn_dp(inp, filter_size, dropout, is_maxpool = True):
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(inp)
    c = BatchNormalization()(c)
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(c)
    c = BatchNormalization()(c)
    if is_maxpool:
        p = MaxPooling2D((2, 2))(c)
        p = Dropout(dropout)(p)
        return c, p
    else:
        c = Dropout(dropout)(c)
        return c

def trans_conv_block_bn_dp(inp, c_inp, filter_size, dropout):
    u = Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(inp)
    u = concatenate([u, c_inp])
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(u)
    c = BatchNormalization()(c)
    c = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(c)
    c = BatchNormalization()(c)
    c = Dropout(dropout)(c)
    return c

def res_trans_conv_block_bn_dp(inp, c_inp, filter_size, dropout):
    u = Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(inp)
    u = BatchNormalization()(u)
    u = concatenate([u, c_inp])
    u = Dropout(dropout)(u)
    return u

def res_identity_block(input_tensor, kernel_size, filters, stage, block, dropout = 0.0):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + 'a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + 'b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + 'c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    return x

def res_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dropout = 0.0):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + 'a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + 'b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + 'c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    x = Dropout(dropout)(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Dropout(dropout)(shortcut)
    shortcut = Activation('relu')(shortcut)

    x = layers.add([x, shortcut])
    return x