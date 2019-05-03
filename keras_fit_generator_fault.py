import keras
import numpy as np
import pytest


class FaultSequence(keras.utils.Sequence):

    def __init__(self):
        self.batch_size = 5
        self.len = 20
        self.X = np.array(range(self.len))
        self.y = np.array(range(self.len)) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        bx = np.zeros((self.batch_size, 1))
        by = np.zeros((self.batch_size, 1))
        #raise RuntimeError()
        return (bx, by)

    def on_epoch_end(self):
        print('epoch done')
        raise RuntimeError()


def simple_net(dense_dim=1):
    inputs = keras.layers.Input(shape=(1,))
    outputs = keras.layers.Dense(1, activation="linear")(inputs)
    net = keras.models.Model(inputs=inputs, outputs=outputs)
    return net


def test_fault_sequence():
    
    inputs = keras.layers.Input(shape=(1,))
    outputs = keras.layers.Dense(1, activation="linear")(inputs)
    net = keras.models.Model(inputs=inputs, outputs=outputs)
    net.compile(optimizer="Adam", loss="mse")

    with pytest.raises(RuntimeError):
        # A RuntimeError is raised as expected but the test never finishes executing.
        net.fit_generator(FaultSequence(), epochs=2, workers=1, use_multiprocessing=False)


if __name__=='__main__':
    test_fault_sequence()
