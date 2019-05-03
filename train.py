import os
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

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
class TensorBoardWithSession(TensorBoard):

    def __init__(self, K,**kwargs):

        self.sess = K.get_session()

        super().__init__(**kwargs)

#TensorBoard = TensorBoardWithSession

from models import build_arch
from data_generators import get_train_val_gen, get_test_data
from knn import knn

def train(config):
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    loss = config["loss"]
    optimizer = config["optimizer"]
    architecture = config["architecture"]
    row_size = config["row_size"]
    col_size = config["col_size"]
    channels = config["channels"]
    num_classes = config["num_classes"]
    snapshot_dir = config["snapshot_dir"]
    log_dir = config["log_dir"]

    #steps_per_epoch = 48000
    optimizer = getattr(optimizers, optimizer)

    model = build_arch(architecture, (row_size,col_size,channels), num_classes)
    model.compile(loss=loss, optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])
    #model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    train_gen, val_data, test_data = get_train_val_gen(batch_size)
    x_test, y_test = test_data

    metadata_path = os.path.join(log_dir,'metadata.tsv')
    if not os.path.exists(metadata_path):
        with open(metadata_path,'w') as fp:
            np.savetxt(fp,np.argmax(y_test, axis=-1))

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_acc',patience=65, restore_best_weights=True, verbose=1))
    callbacks.append(CSVLogger(os.path.join(log_dir,'logs.csv')))
    callbacks.append(ModelCheckpoint(os.path.join(snapshot_dir,"%s_{epoch:02d}-{val_loss:.2f}.h5"%(architecture)),
                                     monitor='val_acc', save_best_only=True, verbose=1,period=10))
    #tb = TensorBoardWithSession(K=K, log_dir=log_dir, write_grads=True, write_images=True,
    #                            embeddings_freq=100, embeddings_layer_names=["features"],
    #                            embeddings_metadata='metadata.tsv', embeddings_data=x_test)
    #tb.set_model(model)
    #callbacks.append(tb)

    #model.load_weights('snapshots/minivgg_best_model.h5')
    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=1000, #len(train_gen),
                        validation_data=val_data,
                        #validation_steps=len(val_gen),
                        callbacks=callbacks,
                        workers=1,
                        verbose=1)

    result = model.evaluate(x_test,y_test)
    print('test loss: %0.4f, test accuracy: %0.4f'%(result[0],result[1]))

    feature_model = Model(model.input, model.layers[-3].output)
    train_gen.reset()
    x_train, y_train = zip(*[next(train_gen) for _ in range(len(train_gen))])
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)

    training_features = feature_model.predict(x_train)

    test_features = feature_model.predict(x_test)

    knn(training_features,y_train, test_features, y_test)


    #model.save(os.path.join(snapshot_dir, architecture+'_best_model.h5'))

@click.command()
@click.option('--config_file')
def main(config_file):
    with open(config_file,'r') as fp:
        config = yaml.load(fp)

    train(config)

if __name__=='__main__':
    main()