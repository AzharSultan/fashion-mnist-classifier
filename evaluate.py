import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import yaml
import click
from keras.models import load_model
from knn import evaluate_knn
from visualize_activation import save_activation_maps
from data_generators import get_fashion_dataset

@click.command()
@click.option('--snapshot')
@click.option('--config_file', default=None)
def evaluate(snapshot, config_file):
    """
    Function to run given model on test data and report the accuracy. It also supports class activation maps
    for example images and testing kNearestNeighbors classification on features extracted from CNN model.
    Args:
        snapshot: path to the saved model snapshot
        config_file: path on yaml config file

    Returns:
        None
    """
    # get parameters
    with open(config_file,'r') as fp:
        config = yaml.load(fp)
    # whether to use kaggle version of test set or keras provided test set
    kaggle_test_set = config["kaggle_test_set"]

    knn_compare = config["knn_compare"]
    activation_maps = config["activation_maps"]
    save_dir = config["save_dir"]

    # load model from snapshot
    model = load_model(snapshot)

    # get fashion-mnist data
    train_gen, val_data, test_data = get_fashion_dataset(32, kaggle_test_set=kaggle_test_set)
    x_test, y_test = test_data
    x_val, y_val = val_data

    # get test and valudation accuracy
    _, accuracy = model.evaluate(x_test,y_test)
    _, val_acc = model.evaluate(x_val, y_val)
    print("Test Accuracy: %0.4f"%(accuracy))
    print("Validation Accuracy: %0.4f"%(val_acc))

    if knn_compare:
        # get kNearestNeighbors accuracy
        knn_accuracy = evaluate_knn(model, train_gen, x_test, y_test)
        print('knn test accuracy: %0.4f'%(knn_accuracy))

    if activation_maps:
        # to visualize what areas in the image model focuses on for prediction
        save_activation_maps(model,x_test,y_test,save_dir)

if __name__=="__main__":
    evaluate()
