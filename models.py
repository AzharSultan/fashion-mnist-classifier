from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, SpatialDropout2D
from keras.initializers import Constant
from keras.regularizers import l2
from wide_resnet import WideResidualNetwork

def minivgg(img_shape, num_classes, weight_decay=0.0, dropout=0.0, spatialdropout=0.0,first_block=32,second_block=64)
    model = Sequential()
    # random erasing
    model.add(SpatialDropout2D(rate=spatialdropout,
                     input_shape=img_shape))
    
    # first conv block
    model.add(Conv2D(first_block, (3, 3), kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Conv2D(first_block, (3, 3), kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout))

    # second conv block
    model.add(Conv2D(second_block, (3, 3), kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Conv2D(second_block, (3, 3), kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=dropout))
    
    # First FC layer
    model.add(Flatten())
    model.add(Dense(512,name="features"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    # Last FC layer + softmax classifier
    # bias is based on class distribution i.e. 10%
    model.add(Dense(num_classes, bias_initializer=Constant(0.2558),activation='softmax'))

    return model

# Not supported currently
def wrn_28_10(img_shape, num_classes):
    model = WideResidualNetwork(28,10,dropout_rate=0.0, include_top=True,weights=None,input_shape=img_shape,
                                classes=num_classes, weight_decay=0.0005)
    return model

# Not supported currently
def wrn_40_4(img_shape, num_classes):
    model = WideResidualNetwork(40,4,dropout_rate=0.4, include_top=True,weights=None,input_shape=img_shape, classes=num_classes)
    return model

def build_arch(architecture, img_shape, num_classes,weight_decay=0.0, dropout=0.0, spatialdropout=0.0,first_block=32,second_block=64):
    model = globals()[architecture](img_shape, num_classes,weight_decay, dropout, spatialdropout,first_block,second_block)
    print(model.summary())
    return model
