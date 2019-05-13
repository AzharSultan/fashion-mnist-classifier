import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10 # needed for one hot encoding
VAL_SIZE = 0.2 # ratio of split between training and validation data
RANDOM_STATE = 1 # for splitting data into training and validation

def create_datagen(train_X):
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        rescale=1./255.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        brightness_range=(0.8,1.2),
        horizontal_flip=True
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

def get_fashion_dataset(batch_size, random_labels=False, kaggle_test_set=False):
    # get data
    (x, y), (x_test,y_test) = fashion_mnist.load_data()

    # if testing kaggle version of fashion-mnist. path is hardcoded
    if kaggle_test_set:
        raw = pd.read_csv('data/fashionmnist/fashion-mnist_test.csv')
        y_test = raw.label
        num_images = raw.shape[0]
        x_as_array = raw.values[:,1:]
        x_test = x_as_array.reshape(num_images, 28, 28)

    # add channels dimension and normalize the data by dividing with 255
    x = np.expand_dims(x,axis=-1)/255
    x_test = np.expand_dims(x_test,axis=-1)/255

    # split data between training and validation
    x_train,x_val, y_train, y_val = train_test_split(x, y, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    # randomize labels if testing convergence of models against random labels
    if random_labels:
        np.random.shuffle(y_train)

    # one-hot incoding
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    # Data augmentation. validation augmentation is mainly about mean subtraction
    train_augmentation = create_datagen(x_train)
    val_augmentation = create_valgen(x_train)

    # create training generator with given batch size
    train_gen = train_augmentation.flow(x_train,y_train, batch_size)

    # create validation and test data after passing it through validation augmentation
    # needed because of a bug in keras
    val_data = next(val_augmentation.flow(x_val,y_val, len(x_val), shuffle=False))
    test_data = next(val_augmentation.flow(x_test,y_test, len(x_test), shuffle=False))

    return train_gen, val_data, test_data

def get_labels():
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    return labels
