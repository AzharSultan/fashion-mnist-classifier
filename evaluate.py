import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import click
from keras.models import load_model
from data_generators import get_train_val_gen

@click.command()
@click.option('--snapshot')
@click.option('--config_file', default=None)
def evaluate(snapshot, config_file):
    model = load_model(snapshot)
    _, val_data, test_data = get_train_val_gen(32)
    x_test, y_test = test_data
    x_val, y_val = val_data
    loss, accuracy = model.evaluate(x_test,y_test)
    _, val_acc = model.evaluate(x_val, y_val)
    print("Test Accuracy: %0.4f"%(accuracy))
    print( "val accuracy: %0.04f"%(val_acc) )

if __name__=="__main__":
    evaluate()
