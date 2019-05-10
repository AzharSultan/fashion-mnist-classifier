import click
from keras.models import load_model
from data_generators import get_train_val_gen

@click.command()
@click.option('--snapshot')
@click.option('--config_file', default=None)
def evaluate(snapshot, config_file):
    model = load_model(snapshot)
    _, _, test_data = get_train_val_gen(32)
    x_test, y_test = test_data

    accuracy = model.evaluate(x_test,y_test)
    print("Test Accuracy: %0.4f"%(accuracy))

if __name__=="__main__":
    evaluate()
