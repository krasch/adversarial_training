import json

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model
from keras import backend as K
import numpy as np
# import scipy.misc

from data import load_mnist, denormalise
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


class FGSM:
    def __init__(self, model):
        self.model = model
        self.gradient_functions = {}

    def generate_adversarial(self, original_image, target_class, epsilon):
        if target_class not in self.gradient_functions:
            self.gradient_functions[target_class] = self._init_gradient_function_(target_class)

        pertubation = np.sign(self.gradient_functions[target_class](original_image))
        adversarial = original_image + epsilon * pertubation
        return np.clip(adversarial, pixel_min, pixel_max)

    def _init_gradient_function_(self, target_class):
        cost_function = self.model.output[0, target_class]
        gradient_function = K.gradients(cost_function, self.model.input)[0]
        func = K.function([self.model.input, K.learning_phase()], [gradient_function])

        def apply(img):
            return func([np.array([img]), 0])[0][0, :]

        return apply


def classify(model, img):
    score = model.predict(np.array([img]))[0]
    idx = np.argmax(score)
    return idx, score[idx]


def test_model_strength(model, attack, test_images, test_labels, epsilon):
    outcome = []

    for img, label in zip(test_images, test_labels):
        for target_label in set(range(10)) - set(label):
            adversarial = attack.generate_adversarial(img, target_label, epsilon)
            new_label, _ = classify(model, adversarial)
            outcome.append(new_label != label)

    print(json.dumps({"epsilon": epsilon, "success_rate": np.array(outcome).mean()}))



def main():
    model = init_mnist_model(pretrained=False)
    attack = FGSM(model)

    _, (X_test, y_test) = load_mnist()
    for epsilon in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        test_model_strength(model, attack, X_test[0:100], y_test[0:100], epsilon=epsilon)

    #for i in range(10):
    #    adversarial = attack.generate_adversarial(X_test[i], target_class=4, epsilon=0.3)
    #    scipy.misc.imsave("generated/adversarial{}.jpg".format(i), denormalise(adversarial))

main()


