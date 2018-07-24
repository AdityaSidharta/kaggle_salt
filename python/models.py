from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, RepeatVector,Reshape
from keras.layers import BatchNormalization, Dropout, Activation
from python.metrics import mean_iou
from python.models_utils import identity_block, conv_block
from keras.models import Model
from python.optim import adam, momentum
from python.config import n_v, n_h
from python.losses import dice_coef_loss

def U_NET():
    input_img = Input((n_h, n_v, 1), name='img')

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer=adam, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model

def U_NET_DEPTH():
    input_img = Input((n_h, n_v, 1), name='img')
    input_features = Input((1,), name='feat')

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    f_repeat = RepeatVector(8 * 8)(input_features)
    f_conv = Reshape((8, 8, 1))(f_repeat)
    p4_feat = concatenate([p4, f_conv], axis = -1)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4_feat)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.2)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img, input_features], outputs=[outputs])
    model.compile(optimizer=adam, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model

def RES_U_NET_DEPTH():
    input_img = Input((n_h, n_v, 1), name='img')

    input_features = Input((1,), name='feat')
    f_repeat = RepeatVector(n_h * n_v)(input_features)
    f_conv = Reshape((n_h, n_v, 1))(f_repeat)

    x = concatenate([input_img, f_conv], axis = -1)

    c1 = conv_block(x, 3, [4, 4, 16], stage=1, block='a', strides=(1,1))
    for block in ['b', 'c', 'd']:
        c1 = identity_block(c1, 3, [4, 4, 16], stage=1, block=block)

    c2 = conv_block(c1, 3, [8, 8, 32], stage=2, block='a')
    for block in ['b', 'c', 'd']:
        c2 = identity_block(c2, 3, [8, 8, 32], stage=2, block=block)

    c3 = conv_block(c2, 3, [16, 16, 64], stage=3, block='a')
    for block in ['b', 'c', 'd']:
        c3 = identity_block(c3, 3, [16, 16, 64], stage=3, block=block)

    c4 = conv_block(c3, 3, [32, 32, 128], stage=4, block='a')
    for block in ['b', 'c', 'd']:
        c4 = identity_block(c4, 3, [32, 32, 128], stage=4, block=block)

    c5 = conv_block(c4, 3, [64, 64, 256], stage = 5, block = 'a')
    for block in [ 'b', 'c', 'd']:
        c5 = identity_block(c5, 3, [64, 64, 256], stage=5, block=block)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = BatchNormalization()(u6)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 3, [32, 32, 128], stage=6, block='a', strides=(1,1))
    for block in ['b', 'c', 'd']:
        c6 = identity_block(c6, 3, [32, 32, 128], stage=6, block=block)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 3, [16, 16, 64], stage=7, block='a', strides=(1,1))
    for block in ['b', 'c', 'd']:
        c7 = identity_block(c7, 3,[16, 16, 64],stage=7, block=block)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 3, [8, 8, 32], stage=8, block='a', strides=(1,1))
    for block in ['b', 'c', 'd']:
        c8 = identity_block(c8, 3, [8, 8, 32], stage=8, block=block)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = BatchNormalization()(u9)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 3, [4, 4, 16], stage=9, block='a', strides=(1,1))
    for block in ['b', 'c', 'd']:
        c9 = identity_block(c9, 3, [4, 4, 16], stage=9, block=block)

    c9 = Dropout(0.2)(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img, input_features], outputs=[outputs])
    model.compile(optimizer=adam, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model