from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, SpatialDropout2D
from keras.initializers import Constant
from wide_resnet import WideResidualNetwork
#from keras.layers import

def minivgg(img_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=img_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.4))

    # minivgg+conv
    # third CONV => RELU => CONV => RELU => POOL layer set
    #model.add(Conv2D(128, (3, 3), padding="valid"))
    #model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    #model.add(Conv2D(128, (3, 3), padding="valid", name="features"))
    #model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    #model.add(MaxPooling2D(pool_size=(3, 3)))
    #model.add(Dropout(rate=0.2))

    # model.add(AveragePooling2D((7,7)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512,name="features"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # softmax classifier
    # bias is based on class distribution i.e. 10%
    model.add(Dense(num_classes, bias_initializer=Constant(0.2558),activation='softmax'))

    return model


def wrn_28_10(img_shape, num_classes):
    model = WideResidualNetwork(28,10,dropout_rate=0.4, include_top=True,weights=None,input_shape=img_shape, classes=num_classes)
    return model

def wrn_40_4(img_shape, num_classes):
    model = WideResidualNetwork(40,4,dropout_rate=0.4, include_top=True,weights=None,input_shape=img_shape, classes=num_classes)
    return model

def build_arch(architecture, img_shape, num_classes):
    model = globals()[architecture](img_shape, num_classes)
    print(model.summary())
    return model