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
from data_generators import get_train_val_gen


def save_embeddings(features, metadata_path, sprite_path, outdir, model_name):
    tf.reset_default_graph()
    embedding_var = tf.Variable(features, name='embeddings')
    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata_path
    embedding.sprite.image_path = sprite_path
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
        summary_writer.close()
        #sess.close()
        #del sess,summary_writer, saver

def save_embeddings_for_training(base_path, outdir):
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    model_paths = glob(base_path+'*.h5')
    print(model_paths)
    sprite_path = os.path.join(outdir,'fashion-mnist-sprite.png')
    metadata_path = os.path.join(outdir,'metadata.tsv')

    _, (x_val, y_val), _ = get_train_val_gen(batch_size=32)
    x_val_images = (x_val - np.min(x_val)) * 255
    y_val = np.argmax(y_val, axis=-1)

    if not os.path.exists(base_path):
        y_str = np.array([labels[j] for j in y_val])
        np.savetxt(metadata_path, y_str, fmt='%s')

    if not os.path.exists(sprite_path):
        plt.imsave(sprite_path, get_sprite_image(x_val_images), cmap='gray')
        exit()

    for model_path in model_paths:
        clear_session()
        model = load_model(model_path)
        model = Model(model.input, model.layers[-3].output)
        features = model.predict(x_val)
        model_id = model_path.split('/')[-1][:-3]
        embedding_dir = os.path.join(outdir,model_id)
        os.makedirs(embedding_dir, exist_ok=True)
        print("passing embeddings to save")
        save_embeddings(features,metadata_path,sprite_path,embedding_dir,model_id)

@click.command()
@click.argument('base_path')
@click.argument('output_dir')
def main(base_path, output_dir):
    save_embeddings_for_training(base_path,output_dir)

if __name__=="__main__":
    main()
