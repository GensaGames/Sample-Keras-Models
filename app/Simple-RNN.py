import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, TimeDistributed
from keras.layers import LSTM
from keras.layers import Activation
np.random.seed(1337)

sample_size = 256
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [1]

x_train = np.array([[x_seed] * sample_size]).reshape(sample_size, len(x_seed), 1)
y_train = np.array([[y_seed] * sample_size]).reshape(sample_size, len(y_seed))
print(x_train.shape)
print(y_train.shape)

model=Sequential()
model.add(SimpleRNN(input_shape=(None, 1), units = 50, return_sequences=True))
model.add(Dense(units = 5, activation  =  "sigmoid"))
model.compile(loss = "mse", optimizer = "rmsprop")
#
# model.fit(x_train, y_train, epochs = 10)

print(model.predict(np.array([[[1],[0],[0],[0],[0],[0]]])).shape)


from keras.utils import plot_model
plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
