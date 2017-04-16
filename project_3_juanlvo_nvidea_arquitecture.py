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
# for line in lines:

	# # center camera
    # source_path = line[0]
    # filename = source_path.split('/')[-1]
    # current_path = 'data/IMG/' + filename
    # #print(current_path)
    # image = cv2.imread(current_path)
	
	# # original center camera
    # images.append(image)
    # measurement = float(line[3])
    # measurements.append(measurement)
	
	# # inverse center camera
    # images.append(cv2.flip(image, 1))
    # measurements.append(measurement*-1.0)  
	
    # correction = float(0.2)
	
	# # left camera
    # source_path = line[1]
    # filename = source_path.split('/')[-1]
    # current_path = 'data/IMG/' + filename
    # image = cv2.imread(current_path)
	
	# # original left camera
    # images.append(image)
    # measurement = float(line[3]) + correction
    # measurements.append(measurement)
	
	# # inverse left camera
    # images.append(cv2.flip(image, 1))
    # measurements.append(measurement*-1.0)	
	
	# # right camera
    # source_path = line[2]
    # filename = source_path.split('/')[-1]
    # current_path = 'data/IMG/' + filename
    # image = cv2.imread(current_path)
	
	# # original right camera
    # images.append(image)
    # measurement = float(line[3]) - correction
    # measurements.append(measurement)
	
	# # inverse right camera
    # images.append(cv2.flip(image, 1))
    # measurements.append(measurement*-1.0)	
	
for line in lines:

	# center camera
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    #print(current_path)
    image_center = cv2.imread(current_path)
	
	# left camera
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image_left = cv2.imread(current_path)	
	
	# right camera
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image_right = cv2.imread(current_path)	
		
    correction = float(0.2)		
		
	# original center camera
    measurement_center = float(line[3])
	
	# original left camera
    measurement_left = float(line[3]) + correction	
	
	# original right camera
    measurement_right = float(line[3]) - correction	

	# add images and angles to data set
    images.extend((image_center, image_left, image_right))
    measurements.extend((measurement_center, measurement_left, measurement_right))	

	# inverse center camera
    images_inverse_center = cv2.flip(image_center, 1)
    measurements_inverse_center = measurement_center*-1.0  

	# inverse left camera
    images_inverse_left = cv2.flip(image_left, 1)
    measurements_inverse_left = measurement_left*-1.0	
	
	# inverse right camera
    images_inverse_right = cv2.flip(image_right, 1)
    measurements_inverse_right = measurement_right*-1.0	
	
	# add images and angles to data set
    images.extend((images_inverse_center, images_inverse_left, images_inverse_right))
    measurements.extend((measurements_inverse_center, measurements_inverse_left, measurements_inverse_right))		

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers.core import Dropout
from keras.layers.advanced_activations import ELU

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(ELU(1.0))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(ELU(1.0))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(ELU(1.0))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))
model.add(ELU(1.0))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))
model.add(ELU(1.0))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
exit()