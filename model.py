import csv
import cv2
import numpy as np

lines = []
		
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)		

images = []
measurements = []	

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.3)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=64):
  num_samples=len(samples)
  correction=float(0.2)
  while True:
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      
      images = []
      measurements = []
      for batch_sample in batch_samples:
        
		#center images
        name_center = 'data/IMG/'+batch_sample[0].split('/')[-1]
        image_center = cv2.imread(name_center)
        measurement_center = float(batch_sample[3])
        
        images.append(image_center)
        measurements.append(measurement_center)
		
		#left images
        name_left = 'data/IMG/'+batch_sample[1].split('/')[-1]
        image_left = cv2.imread(name_left)
        measurement_left = float(batch_sample[3]) + correction
		
        images.append(image_left)
        measurements.append(measurement_left)
        
		#right images
        name_right = 'data/IMG/'+batch_sample[2].split('/')[-1]
        image_right = cv2.imread(name_right)
        measurement_right = float(batch_sample[3]) - correction				
        
        images.append(image_right)
        measurements.append(measurement_right)			
        
        # inverse center camera
        images_inverse_center = cv2.flip(image_center, 1)
        measurements_inverse_center = measurement_center*-1.0  
		
        images.append(images_inverse_center)
        measurements.append(measurements_inverse_center)
        
        # inverse left camera
        images_inverse_left = cv2.flip(image_left, 1)
        measurements_inverse_left = measurement_left*-1.0

        images.append(images_inverse_left)
        measurements.append(measurements_inverse_left)
        
        # inverse right camera
        images_inverse_right = cv2.flip(image_right, 1)
        measurements_inverse_right = measurement_right*-1.0
		
        images.append(images_inverse_right)
        measurements.append(measurements_inverse_right)							

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
print(len(train_samples))
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers.core import Dropout
from keras.layers.advanced_activations import ELU
import matplotlib.pyplot as plt

#########################################################################
#
# Model NVIDEA
#
#########################################################################

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.59))
model.add(ELU(1.0))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.49))
model.add(ELU(1.0))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.39))
model.add(ELU(1.0))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.29))
model.add(ELU(1.0))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.19))
model.add(ELU(1.0))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=30, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.summary()
model.get_config()

exit()