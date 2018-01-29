from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from data import load_mnist
from util import JSONLogger


def make_model():
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

    return model


(X_train, y_train), (X_test, y_test) = load_mnist()
model = make_model()
model.fit(X_train, y_train, batch_size=64, epochs=20,
          validation_data=(X_test, y_test),
          callbacks=[JSONLogger()], verbose=0)