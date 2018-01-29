from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model
from keras import backend as K
import numpy as np

from data import load_mnist
from util import JSONLogger

pixel_min = 0.0
pixel_max = 1.0


def init_mnist_model(pretrained=False):
    if pretrained:
        return load_model("mnist_model.h5")

    else:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        (X_train, y_train), (X_test, y_test) = load_mnist()
        model.fit(X_train, y_train, batch_size=64, epochs=2,
                  validation_data=(X_test, y_test),
                  callbacks=[JSONLogger()], verbose=0)

        return model


def init_get_gradient(model):
    get_gradient_functions = {}
    for target_class in range(10):
        cost_function = model.output[0, target_class]
        gradient_function = K.gradients(cost_function, model.input)[0]
        get_gradient_functions[target_class] = K.function([model.input, K.learning_phase()], [gradient_function])

    def apply(img, target_class):
        func = get_gradient_functions[target_class]
        return func([np.array([img]), 0])[0][0, :]

    return apply


def FGSM(original_image, get_gradient, target_class, epsilon):
    pertubation = np.sign(get_gradient(original_image, target_class))
    adversarial = original_image + epsilon * pertubation
    return np.clip(adversarial, pixel_min, pixel_max)


def main():
    model = init_mnist_model(pretrained=False)
    """
    get_gradient = init_get_gradient(model)

    _, (X_test, y_test) = load_mnist()

    for i in range(10):
        adversarial = FGSM(X_test[i], get_gradient, 4, epsilon=0.3)

        scipy.misc.imsave("zebra_adversarial2.jpg", adversarial)
    """

main()


