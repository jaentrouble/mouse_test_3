import tensorflow as tf
from tensorflow import keras

inputs = keras.Input((80,80,1))
x = keras.layers.Conv2D(filters = 32, kernel_size = (8,8),strides = 4, activation= 'relu')(inputs)
x = keras.layers.Conv2D(filters = 64, kernel_size = (4,4),strides = 2, activation= 'relu')(x)
x = keras.layers.Conv2D(filters = 64, kernel_size = (3,3),strides = 1, activation= 'relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units = 512, activation = 'relu')(x)
outputs = keras.layers.Dense(6)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()