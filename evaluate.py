import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import yaml
import click
from keras.models import load_model
from knn import evaluate_knn
from visualize_activation import save_activation_maps
from data_generators import get_train_val_gen

@click.command()
@click.option('--snapshot')
@click.option('--config_file', default=None)
def evaluate(snapshot, config_file):
    with open(config_file,'r') as fp:
        config = yaml.load(fp)
    knn_compare = config["knn_compare"]
    activation_maps = config["activation_maps"]
    save_dir = config["save_dir"]

    model = load_model(snapshot)
    train_gen, val_data, test_data = get_train_val_gen(32)
    x_test, y_test = test_data
    x_val, y_val = val_data
    _, accuracy = model.evaluate(x_test,y_test)
    _, val_acc = model.evaluate(x_val, y_val)
    print("Test Accuracy: %0.4f"%(accuracy))
    print("Validation Accuracy: %0.04f"%(val_acc))

    if knn_compare:
        knn_accuracy = evaluate_knn(model, train_gen, x_test, y_test)
        print('knn test accuracy: %0.4f'%(knn_accuracy))

    if activation_maps:
        save_activation_maps(model,x_test,y_test,save_dir)

if __name__=="__main__":
    evaluate()
