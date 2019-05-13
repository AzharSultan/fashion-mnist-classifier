import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import click
import yaml
from skimage.io import imread
from keras.models import load_model

@click.command()
@click.option('--snapshot')
@click.option('--image')
@click.option('--config_file', default=None)
def test(snapshot, image, config_file):
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    with open(config_file,'r') as fp:
        config = yaml.load(fp)
    mean = config["mean"]
    model = load_model(snapshot)
    img = imread(image)
    img = img/255.0 - mean
    img = img.reshape((28,28,1))
    prediction = model.predict(img)
    print("Sample %s: class %s"%(image, labels[prediction]))

if __name__=="__main__":
    test()