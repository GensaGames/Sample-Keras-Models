import numpy as np
from keras.activations import relu
from keras.losses import MSE
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.utils import plot_model

#                       samples, timeline, image_x, image_y, channels
x_train = np.random.rand(25,       10,       75,     75,       1)
y_train = np.random.randint(0, 2, size=(25, 1))

cnn = Sequential()
cnn.add(Conv2D(
    filters=1, kernel_size=(10, 10), activation=relu,
    padding='same', input_shape=(75, 75, 1), data_format='channels_last'))

cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())

model = Sequential()
model.add(TimeDistributed(cnn))
model.add(LSTM(12))
model.add(Dense(1))



model.compile(loss=MSE, optimizer=RMSprop())
plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

model.fit(x_train, y_train, epochs=30)
