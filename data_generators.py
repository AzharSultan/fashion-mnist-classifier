import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10 # needed for one hot encoding
TEST_SIZE = 0.2 # ratio of split between training and validation data
RANDOM_STATE = 0 # for splitting data into training and validation

def create_datagen(train_X):
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        rescale=1./255.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        brightness_range=(0.8,1.2),
        horizontal_flip=True,
        #vertical_flip=True
    )
    data_generator.fit(train_X)
    return data_generator
def create_valgen(train_X):
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        rescale=1./255.,
        brightness_range=(1.0,1.0), # needed due to a possible bug in keras implementation
    )
    data_generator.fit(train_X)
    return data_generator

def get_train_val_gen(batch_size, random_labels=False):
    (x, y), (x_test,y_test) = fashion_mnist.load_data()
    x = np.expand_dims(x,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)/255
    x_train,x_val, y_train, y_val = train_test_split(x/255,y,test_size=0.2,random_state=1)
    if random_labels:
        np.random.shuffle(y_train)
    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    train_augmentation = create_datagen(x_train)
    val_augmentation = create_valgen(x_train)

    train_gen = train_augmentation.flow(x_train,y_train, batch_size)
    val_data = next(val_augmentation.flow(x_val,y_val, len(x_val), shuffle=False))
    test_data = next(val_augmentation.flow(x_test,y_test, len(x_test), shuffle=False))

    #val_data = (x_val-train_augmentation.mean, y_val)
    #test_data = (x_test-train_augmentation.mean, y_test)

    return train_gen, val_data, test_data

def get_test_gen(batch_size):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)
    y_test = to_categorical(y_test, num_classes=10)
    test_augmentation = create_valgen(x_train)
    test_gen = test_augmentation.flow(x_test, y_test, batch_size)
    return test_gen

def get_test_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test = np.expand_dims(x_test, axis=-1)/255
    return x_test, y_test
