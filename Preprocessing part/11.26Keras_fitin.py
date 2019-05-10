# import part
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

# download data
clean_mask = np.loadtxt("clean_mask.txt")
spec_train = np.loadtxt("spec_train.txt")
clean_mask_test = np.loadtxt("clean_mask_test.txt")
spec_train_test = np.loadtxt("spec_train_test.txt")

# data pre-processing
X_train = spec_train
X_test = clean_mask
y_train = spec_train_test
y_test = clean_mask_test

# build NN
model = Sequential([
    Dense(2470, input_dim=2470),
    Activation('sigmoid')#,
    #Dense(2470),
    #Activation('softmax')
    ])

# optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(
    optimizer= rmsprop,
    loss = 'mean_squared_error',
    metrics=['accuracy']#, mean_pred]
    )

print('Train-------------------------')
'''
# training
model.fit(X_train, y_train, nb_epoch=20, batch_size=20)

print('\nTrain~~~~~~~~~~~~~~~~')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss', loss)
print('test accuracy', accuracy)

# layer_name = 'my_layer'
# intermediate_layer_model = model(input=model.input,
#                                  output=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)



# Sequential模型
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([spec_train])[0]

print(layer_output)

'''

history = model.fit(X_train, y_train, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([spec_train])[0]

print(layer_output)