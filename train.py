import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import click
import yaml
import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
from keras import optimizers
from keras.models import Model,load_model
from keras import backend as K
#import tensorflow
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from models import build_arch
from data_generators import get_train_val_gen
from visualize_activation import save_activation_maps
from knn import evaluate_knn

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def train(config):


    epochs = config["epochs"]
    loss = config["loss"]
    architecture = config["architecture"]
    row_size = config["row_size"]
    col_size = config["col_size"]
    channels = config["channels"]
    num_classes = config["num_classes"]
    snapshot_dir = config["snapshot_dir"]
    log_dir = config["log_dir"]
    knn_compare = config["knn_compare"]
    activation_maps = config["activation_maps"]
    hyper_optimization = config["hyper_optimization"]
    random_labels = config["random_labels"]
    initial_epoch = config["initial_epoch"]
    starting_checkpoint = config["starting_checkpoint"]


    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    #optimizer = {{choice(['Adam','SGD'])}}
    if hyper_optimization:
        learning_rate = {{uniform(0.1,0.00001)}}
        batch_size = {{choice([64,128,256])}}
        optimizer = {{choice(['adam', 'sgd'])}}
        weight_decay = {{uniform(0.,0.0001)}}
        dropout = {{uniform(0.,0.7)}}
        spatialdropout = {{uniform(0.,0.4)}}
        first_block = {{choice([32,64])}}
    else:
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        optimizer = config["optimizer"]
        weight_decay = config["weight_decay"]
        dropout = config["dropout"]
        spatialdropout = config["spatialdropout"]
        first_block = config["first_block"]

    second_block = 2*first_block

    if optimizer == 'Adam':
        opt = optimizers.Adam(lr=learning_rate)
    else:
        opt = optimizers.SGD(lr=learning_rate, momentum=0.9)

    img_shape = row_size,col_size,channels
    model = build_arch(architecture, img_shape, num_classes,weight_decay, dropout,
                       spatialdropout, first_block, second_block)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    train_gen, val_data, test_data = get_train_val_gen(batch_size, random_labels)
    x_test, y_test = test_data

    callbacks = []
    basename = "%s_lr%0.4f_bs%d_%s_wd%0.5f_do%0.2f_sdo%0.2f_fb%d.csv" % \
               (architecture,learning_rate, batch_size, optimizer, weight_decay, dropout, spatialdropout, first_block)
    callbacks.append(EarlyStopping(monitor='val_acc',patience=75, restore_best_weights=True, verbose=1))
    callbacks.append(CSVLogger(os.path.join(log_dir,basename)))
    model_checkpoint_path = os.path.join(snapshot_dir,"%s_{epoch:02d}-{val_loss:.2f}.h5"%(basename))
    callbacks.append(ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=1,period=1))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',patience=50, factor=0.2, min_lr=0.0000001, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor='loss',patience=15, factor=0.2, min_lr=0.0001, verbose=1))

    if starting_checkpoint:
        model.load_weights(starting_checkpoint)

    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=1000, # as asked in problem statement
                        validation_data=val_data,
                        callbacks=callbacks,
                        workers=1,
                        initial_epoch=initial_epoch,
                        verbose=2)

    result = model.evaluate(x_test,y_test)
    print('test loss: %0.4f, test accuracy: %0.4f'%(result[0],result[1]))

    model.save(os.path.join(snapshot_dir, architecture+'_best_model.h5'))
    return {'loss': -result[1], 'status': STATUS_OK, 'model': model}

def data():
    config_file = 'config/train.yml'
    with open(config_file,'r') as fp:
        config = yaml.load(fp)
    return config

@click.command()
@click.option('--config_file')
def main(config_file):
    with open(config_file,'r') as fp:
        config = yaml.load(fp)

    #train(config)
    hyper_optimization = config['hyper_optimization']
    if not hyper_optimization:
        train(config)
    else:
        best_run, best_model = optim.minimize(model=train,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=20,
                                              trials=Trials())
        print("Best performing model chosen hyper-parameters:")
        print(best_run)

if __name__=='__main__':
    main()
