from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, RepeatVector,Reshape
from keras.layers import BatchNormalization, Dropout, Activation, Flatten, Dense
from python.metrics import mean_iou
from python.models_utils import res_identity_block, res_conv_block, conv_block, \
    conv_block_bn_dp, trans_conv_block, trans_conv_block_bn_dp, res_trans_conv_block_bn_dp, \
    dense_block_bn_dp
from keras.models import Model
from python.optim import adam, momentum, adamw
from python.config import n_v, n_h
from python.losses import dice_coef_loss
from fastkeras.optim import AdamW

def U_NET():
    input_img = Input((n_h, n_v, 1), name='img')
    c1, p1 = conv_block(input_img, 8)
    c2, p2 = conv_block(p1, 16)
    c3, p3 = conv_block(p2, 32)
    c4, p4 = conv_block(p3, 64)
    c5 = conv_block(p4, 128, is_maxpool= False)
    c6 = trans_conv_block(c5, c4, 64)
    c7 = trans_conv_block(c6, c3, 32)
    c8 = trans_conv_block(c7, c2, 16)
    c9 = trans_conv_block(c8, c1, 8)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer=adam, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model

def U_NET_DEPTH():
    input_img = Input((n_h, n_v, 1), name='img')
    input_features = Input((1,), name='feat')
    c1, p1 = conv_block_bn_dp(input_img, 8, 0.25)
    c2, p2 = conv_block_bn_dp(p1, 16, 0.25)
    c3, p3 = conv_block_bn_dp(p2, 32, 0.25)
    c4, p4 = conv_block_bn_dp(p3, 64, 0.25)

    f_repeat = RepeatVector(8 * 8)(input_features)
    f_conv = Reshape((8, 8, 1))(f_repeat)
    p4_feat = concatenate([p4, f_conv], axis = -1)

    c5 = conv_block_bn_dp(p4_feat, 128, 0.25, is_maxpool = False)
    c6 = trans_conv_block_bn_dp(c5, c4, 64, 0.25)
    c7 = trans_conv_block_bn_dp(c6, c3, 32, 0.25)
    c8 = trans_conv_block_bn_dp(c7, c2, 16, 0.25)
    c9 = trans_conv_block_bn_dp(c8, c1, 8, 0.25)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img, input_features], outputs=[outputs])
    model.compile(optimizer=adamw, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model

def U_NET_DEPTH_CLASSIFIER():
    input_img = Input((n_h, n_v, 1), name ='img')
    input_features = Input((1, ), name = 'feat')
    c1, p1 = conv_block_bn_dp(input_img, 8, 0.25)
    c2, p2 = conv_block_bn_dp(p1, 16, 0.25)
    c3, p3 = conv_block_bn_dp(p2, 32, 0.25)
    c4, p4 = conv_block_bn_dp(p3, 64, 0.25)

    p4_flat = Flatten()(p4)
    p4_feat = concatenate([p4_flat, input_features])
    classifier_output = dense_block_bn_dp(p4_feat, [32, 1], dropout=0.25)

    f_repeat = RepeatVector(8 * 8)(input_features)
    f_conv = Reshape((8, 8, 1))(f_repeat)
    p4_concat = concatenate([p4, f_conv], axis=-1)

    c5 = conv_block_bn_dp(p4_concat, 128, 0.25, is_maxpool = False)
    c6 = trans_conv_block_bn_dp(c5, c4, 64, 0.25)
    c7 = trans_conv_block_bn_dp(c6, c3, 32, 0.25)
    c8 = trans_conv_block_bn_dp(c7, c2, 16, 0.25)
    c9 = trans_conv_block_bn_dp(c8, c1, 8, 0.25)
    mask_outputs = Conv2D(1, (1, 1), activation='sigmoid', name='mask')(c9)

    model = Model(inputs={input_img, input_features}, outputs=[classifier_output, mask_outputs])
    model.compile(optimizer=adamw, loss=['binary_crossentropy', dice_coef_loss], metrics=[mean_iou, 'accuracy'])
    return model

def RES_U_NET_DEPTH():
    input_img = Input((n_h, n_v, 1), name='img')

    input_features = Input((1,), name='feat')
    f_repeat = RepeatVector(n_h * n_v)(input_features)
    f_conv = Reshape((n_h, n_v, 1))(f_repeat)

    x = concatenate([input_img, f_conv], axis = -1)

    c1 = res_conv_block(x, 3, [4, 4, 16], stage=1, block='a', strides=(1, 1))
    c1 = Dropout(0.5)(c1)
    for block in ['b', 'c', 'd']:
        c1 = res_identity_block(c1, 3, [4, 4, 16], stage=1, block=block)
    c1 = Dropout(0.5)(c1)

    c2 = res_conv_block(c1, 3, [8, 8, 32], stage=2, block='a')
    c2 = Dropout(0.5)(c2)
    for block in ['b', 'c', 'd']:
        c2 = res_identity_block(c2, 3, [8, 8, 32], stage=2, block=block)
    c2 = Dropout(0.5)(c2)

    c3 = res_conv_block(c2, 3, [16, 16, 64], stage=3, block='a')
    c3 = Dropout(0.5)(c3)
    for block in ['b', 'c', 'd']:
        c3 = res_identity_block(c3, 3, [16, 16, 64], stage=3, block=block)
    c3 = Dropout(0.5)(c3)

    c4 = res_conv_block(c3, 3, [32, 32, 128], stage=4, block='a')
    c4 = Dropout(0.5)(c4)
    for block in ['b', 'c', 'd']:
        c4 = res_identity_block(c4, 3, [32, 32, 128], stage=4, block=block)
    c4 = Dropout(0.5)(c4)

    c5 = res_conv_block(c4, 3, [64, 64, 256], stage = 5, block ='a')
    c5 = Dropout(0.5)(c5)
    for block in [ 'b', 'c', 'd']:
        c5 = res_identity_block(c5, 3, [64, 64, 256], stage=5, block=block)
    c5 = Dropout(0.5)(c5)

    u6 = res_trans_conv_block_bn_dp(c5, c4, 128, 0.5)
    c6 = res_conv_block(u6, 3, [32, 32, 128], stage=6, block='a', strides=(1, 1))
    for block in ['b', 'c', 'd']:
        c6 = res_identity_block(c6, 3, [32, 32, 128], stage=6, block=block)
    c6 = Dropout(0.5)(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.5)(u7)
    c7 = res_conv_block(u7, 3, [16, 16, 64], stage=7, block='a', strides=(1, 1))
    for block in ['b', 'c', 'd']:
        c7 = res_identity_block(c7, 3, [16, 16, 64], stage=7, block=block)
    c7 = Dropout(0.5)(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.5)(u8)
    c8 = res_conv_block(u8, 3, [8, 8, 32], stage=8, block='a', strides=(1, 1))
    for block in ['b', 'c', 'd']:
        c8 = res_identity_block(c8, 3, [8, 8, 32], stage=8, block=block)
    c8 = Dropout(0.5)(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = BatchNormalization()(u9)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.5)(u9)
    c9 = res_conv_block(u9, 3, [4, 4, 16], stage=9, block='a', strides=(1, 1))
    for block in ['b', 'c', 'd']:
        c9 = res_identity_block(c9, 3, [4, 4, 16], stage=9, block=block)
    c9 = Dropout(0.5)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img, input_features], outputs=[outputs])
    model.compile(optimizer=adamw, loss= dice_coef_loss, metrics = [mean_iou, 'accuracy'])
    return model