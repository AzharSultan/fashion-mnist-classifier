import os
import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
from keras.models import load_model, Model
from keras.backend import clear_session
from tensorflow.contrib.tensorboard.plugins import projector

from helpers import get_sprite_image
from data_generators import get_fashion_dataset, get_labels


def save_embeddings(features,outdir, model_name):
    """
    Fuction to save embeddings data for tensorboard
    Args:
        features: data to use as embedding
        outdir: output directory
        model_name: name of the embeddings file being saved

    Returns:

    """
    tf.reset_default_graph()
    embedding_var = tf.Variable(features, name='embeddings')
    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    # path to theses files needs to be relative to the output directory
    embedding.metadata_path = '../metadata.tsv'
    embedding.sprite.image_path = '../fashion-mnist-sprite.png'
    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([28, 28])

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(outdir)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outdir, '%s.ckpt'%(model_name)))

def save_embeddings_for_training(base_path, outdir):
    """
    Function that searches for all model snapshot in base_path and creates embeddings for them. It extracts features
    from third last layer of the model for embeddings.
    Args:
        base_path: Directory where to look for model snapshots
        outdir: output directory where embeddings are saved. Embeddings for each model are saved in a separate folder

    Returns:
        None
    """
    # look for model snapshots in the base_path
    model_paths = glob(base_path+'*.h5')

    sprite_path = os.path.join(outdir,'fashion-mnist-sprite.png')
    metadata_path = os.path.join(outdir,'metadata.tsv')

    # get fashion-mnist data
    _, (x_val, y_val), _ = get_fashion_dataset(batch_size=32)
    # convert normalized images back to 0-255 range (to create sprite image)
    x_val_images = (x_val - np.min(x_val)) * 255
    # convert one-hot-encoded labels back to values
    y_val = np.argmax(y_val, axis=-1)

    if not os.path.exists(base_path):
        # save labels as class names in metadata
        labels = get_labels()
        y_str = np.array([labels[j] for j in y_val])
        np.savetxt(metadata_path, y_str, fmt='%s')

    if not os.path.exists(sprite_path):
        # save sprite image
        plt.imsave(sprite_path, get_sprite_image(x_val_images), cmap='gray')
        exit()

    # save embeddings in a separate folder for each path
    for model_path in model_paths:
        # needed to avoided retaining previous embeddings data in the for loop
        clear_session()

        # create new folder based on model snapshot name
        model_id = model_path.split('/')[-1][:-3]
        embedding_dir = os.path.join(outdir, model_id)
        os.makedirs(embedding_dir, exist_ok=True)

        # extract features from third last layer of the model
        model = load_model(model_path)
        model = Model(model.input, model.layers[-3].output)
        features = model.predict(x_val)

        save_embeddings(features,embedding_dir,model_id)

@click.command()
@click.argument('base_path')
@click.argument('output_dir')
def main(base_path, output_dir):
    save_embeddings_for_training(base_path,output_dir)

if __name__=="__main__":
    main()
