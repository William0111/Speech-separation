# 0131_RNN from Movan, to simulate the minst RNN


import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import RMSprop

TIME_STEPS = 3
INPUT_SIZE = 512
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 1536
CELL_SIZE = 1200
LR = 0.001


           
           
# loading data
X_train = np.loadtxt("0126_spec_train_1000.txt")
y_train = np.loadtxt("0126_mask_train_1000.txt")

X_test = np.loadtxt("0126_spec_test_1000.txt")
y_test = np.loadtxt("0126_mask_test_1000.txt")

# batch_size=y_test.shape[0]

# data pre-processing
X_train = X_train.reshape(-1, 3, 512)
#y_train = y_train.reshape(-1, 3, 512)
X_test = X_test.reshape(-1, 3, 512)
#y_test = y_test.reshape(-1, 3, 512)

#  build RNN model
model = Sequential()

# RNN cell

model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    # input_dim=INPUT_SIZE,
    # input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True
))

# output layer
#model.add(Flatten())
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('hard_sigmoid'))

# # optimizer
#
adam = Adam(LR)

#optimizer
#rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # training
#

for step in range(20001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        print('Next_Train----------: step = ', step)
        train_cost, train_accuracy = model.evaluate(X_train, y_train, batch_size=y_train.shape[0], verbose=False)
        print('train_cost: ', train_cost, 'train_accuracy: ', train_accuracy)
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
        
       

# print('Train-------------------------')
#
# history = model.fit(X_train, y_train, validation_split=0.25, epochs=3100, shuffle=True, batch_size=75, verbose=1)
#
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# get_3rd_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[1].output])
# layer_output1 = get_3rd_layer_output([X_train])[0]
#
# print(layer_output1)
