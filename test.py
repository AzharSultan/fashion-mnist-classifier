import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import click
import yaml
import numpy as np
from skimage.io import imread
from keras.models import load_model
from data_generators import get_labels

@click.command()
@click.option('--snapshot')
@click.option('--image')
@click.option('--config_file', default=None)
def test(snapshot, image, config_file):
    """
    Function that runs the given model on the provided image and prints the name of predicted class
    Args:
        snapshot: path to the saved model snapshot
        image: path to the test image
        config_file: path to configuration file

    Returns:
        None
    """
    labels = get_labels()
    with open(config_file,'r') as fp:
        config = yaml.load(fp)
    mean = config["mean"]
    model = load_model(snapshot)
    img = imread(image)
    img = img/255.0 - mean
    img = img.reshape((1,28,28,1))
    prediction = model.predict(img)
    prediction = np.argmax(prediction[0], axis=-1)
    print("Sample %s: class %s"%(image, labels[prediction]))

if __name__=="__main__":
    test()
