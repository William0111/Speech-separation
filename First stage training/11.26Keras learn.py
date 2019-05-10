# import part
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.
X_test = X_test.reshape(X_test.shape[0], -1) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build NN
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
    ])

# optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)


model.compile(
    optimizer= rmsprop,
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
    )

print('Train-------------------------')

# training
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

print('\nTrain-------------------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss', loss)
print('test accuracy', accuracy)