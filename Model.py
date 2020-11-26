import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

path1 = 'Resources/dataset/Training'
path2 = 'Resources/dataset/Test'

mask = Sequential()
mask.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
mask.add(BatchNormalization())
mask.add(MaxPooling2D(pool_size = (2, 2),padding='same'))


mask.add(Convolution2D(32, 3, 3, activation = 'relu'))
mask.add(BatchNormalization())
mask.add(MaxPooling2D(pool_size = (2, 2),padding='same'))


mask.add(layers.Flatten())
mask.add(layers.Dense(100, activation='relu'))
mask.add(layers.Dense(1, activation='sigmoid'))

mask.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(path1,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(path2,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

mask.fit_generator(training_set,
                         steps_per_epoch = 90,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 25)
mask.summary()
mask.save("Mask_model")
