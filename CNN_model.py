# Import the required libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Load the data
mnist = keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#expand 1 more dimention as 1 for colour channel gray
train_X = train_X.reshape(train_X.shape[0], 28, 28,1)
test_X = test_X.reshape(test_X.shape[0], 28, 28,1)

# Preprocessing - Z-score
mean_px = train_X.mean().astype(np.float32)
std_px = train_X.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

# Define the model
seed = 43
np.random.seed(seed)

from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

model = Sequential([
    Lambda(standardize, input_shape=(28,28,1)),
    Convolution2D(32,(3,3), activation='relu'),
    Convolution2D(32,(3,3), activation='relu'),
    MaxPooling2D(),
    Convolution2D(64,(3,3), activation='relu'),
    Convolution2D(64,(3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
    ])

# Compile the model
model.compile(Adam(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Split the data
X = train_X
y = train_y
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.10, random_state=42)

# Create a log directory 
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# Run the model
from keras.callbacks import EarlyStopping

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=512,
    epochs=100,
    callbacks=[early_stopping, tensorboard_cb])

# Make predictions on test data
from sklearn.metrics import accuracy_score

predictions = model.predict(test_X)

# Evaluate the model
score = accuracy_score(test_y, np.argmax(predictions,axis=1))
print('MAE:', score)

