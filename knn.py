import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def knn(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    model.fit(x_train,y_train)

    result = model.score(x_test, y_test)

    print("knn accuracy: %0.4f"%(result))