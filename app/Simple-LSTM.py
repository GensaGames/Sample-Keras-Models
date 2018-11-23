
from keras.models import Model
from keras.layers import Input, InputLayer
from keras.layers import LSTM
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, TimeDistributed
from keras.layers import LSTM
from keras.layers import Activation

# define model
model = Sequential()
model.add(InputLayer(input_shape=(3, 1)))
model.add(LSTM(units=1, return_sequences=True))

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))

# make and show prediction
print(model.predict(data).shape)