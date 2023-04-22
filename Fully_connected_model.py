# Import the required libraries
import numpy as np # linear algebra
import pandas as pd 
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

# Preprocessing - Feature standardization
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
    Lambda(standardize,input_shape=(28,28)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
    ])

# Compile the model
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])

# Run the model
from sklearn.model_selection import train_test_split

X = train_X
y = train_y
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.10, random_state=42)

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],verbose=1)

# Make predictions on test data
from sklearn.metrics import accuracy_score

predictions = model.predict(test_X)

score = accuracy_score(test_y, np.argmax(predictions,axis=1))
print('Accuracy:', score)
