from keras import optimizers
from fastkeras import optim

adam = optimizers.adam(lr=1e-3, amsgrad=True)
momentum = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov= True)
adamw = optim.AdamW(lr=1e-3)