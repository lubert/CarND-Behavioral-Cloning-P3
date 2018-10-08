### Read CSV
import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip the first line
    for line in reader:
        samples.append(line)
        
### Define test and validation generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

img_path = 'data/IMG/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Randomly select a camera
                angle = float(batch_sample[3])
                camera = np.random.randint(0, 3)
                name = img_path + batch_sample[camera].split('/')[-1]
                image = cv2.imread(name)
                # Convert to YUV, per NVIDIA architecture
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                correction = 0.2 # Steering correction factor
                # Left
                if camera == 1:
                    angle = angle + correction
                # Right
                elif camera == 2:
                    angle = angle - correction
                # Randomly flip the image horizontally
                flip = np.random.randint(0, 2)
                if flip == 1:
                    image = np.fliplr(image)
                    angle = -angle
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

### Setup Keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

filepath="model-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=20, callbacks=[checkpoint])