import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Model

def evaluate_knn(model, train_gen, x_test, y_test):
    """

    Args:
        model: keras model where third last layer is the layer from which to extract features
        train_gen: generator to get all training data
        x_test: test data
        y_test: test data

    Returns:
        result: accuracy of knn classifier on test data
    """
    feature_model = Model(model.input, model.layers[-3].output)
    train_gen.reset()
    x_train, y_train = zip(*[next(train_gen) for _ in range(len(train_gen))])
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)

    training_features = feature_model.predict(x_train)

    test_features = feature_model.predict(x_test)

    result = knn(training_features,y_train, test_features, y_test)
    return result

def knn(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    model.fit(x_train,y_train)

    result = model.score(x_test, y_test)

    return result