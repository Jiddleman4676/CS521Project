import numpy as np
from sklearn.model_selection import train_test_split

# store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    x = np.clip(x, -100, 100) # prevent weights from overflowing
    x += + 1e-10 # prevent divide by 0
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

# loss function
def NNN_loss(y_pred, y_true, derivative=False):
    if derivative:
        grad = np.copy(y_pred)
        grad[range(len(y_pred)), y_true] -= 1
        return grad / len(y_pred)

    return np.mean(-y_pred[range(len(y_pred)), y_true])

def balanced_train_test_split(X, y, test_size, random_state=None):
    test_size = 1 - test_size
    classes = [1, 2, 0]
    X_train, X_test = [], []
    y_train, y_test = [], []
    buffer_0 = 0
    buffer_1 = 0
    index = 0
    # iterate through classes in order of size of the class [1,2,0]
    for cls in classes:
        # class 1: prediabetes
        if index == 0:
            class_size = int(test_size * (1/3)*len(y))
        
        # class 2: diabetes
        if index == 1 and buffer_0 != 0:
            class_size = int(test_size * (1/3)*len(y) + (buffer_0 / 2))
        else:
            class_size = int(test_size * (1/3)*len(y))
        
        # class 0: no diabetes
        if index == 2 and buffer_0 != 0 and buffer_1 != 0:
            class_size = int(test_size * (1/3)*len(y) + buffer_1)
        elif index == 2 and buffer_0 != 0:
            class_size = int(test_size * (1/3)*len(y) + (buffer_0 / 2))
        else:
            class_size = int(test_size * (1/3)*len(y))
        index = index + 1

        cls_indices = np.where(y == cls)[0]
        cls_X = X[cls_indices]
        cls_y = y[cls_indices]
        max_train_class_size = int(.8 * len(cls_y))
        if max_train_class_size < class_size:
            if index == 0:
                buffer_0 = class_size - max_train_class_size
            else:
                buffer_1 = class_size - max_train_class_size
            class_size = max_train_class_size
            X_cls_train = cls_X
            y_cls_train = cls_y
        
        # train_test_split for the current class
        X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
                cls_X, cls_y, train_size=class_size, random_state=random_state
        )

        # combine all classes
        X_train.append(X_cls_train)
        X_test.append(X_cls_test)
        y_train.append(y_cls_train)
        y_test.append(y_cls_test)

    # combine all classes
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    return X_train, X_test, y_train, y_test
