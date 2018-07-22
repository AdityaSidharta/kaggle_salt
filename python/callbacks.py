from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from fastkeras.callbacks import SGDRScheduler
from python.config import max_lr, min_lr, steps_per_epoch, lr_decay, cycle_length, mult_factor


def get_callbacks(model_name):
    callbacks = [
        SGDRScheduler(min_lr = min_lr, max_lr = max_lr, steps_per_epoch=steps_per_epoch, lr_decay=lr_decay,
                      cycle_length=cycle_length, mult_factor=mult_factor),
        EarlyStopping(monitor='loss', patience=5, verbose=1),
        TensorBoard(log_dir='./logs', write_graph=True, write_images=True),
        ModelCheckpoint(monitor='loss', filepath='./model/' + model_name +'.h5', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    return callbacks