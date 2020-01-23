import tensorflow as tf
# Building the CNN

# Importing the Keras libraries and packages'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.summary()
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/cnn_imgs_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/cnn_imgs_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# this will take long time
classifier.fit_generator(training_set,
                         samples_per_epoch = 10,#samples_per_epoch = 8000,
                         nb_epoch = 25, # nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 20)#nb_val_samples = 2000)
#UnboundLocalError: local variable 'epoch_logs' referenced before assignment

'''
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5, # nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
'''



import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/cnn_imgs_dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# This is test img
# first arg is the path
# img is 64x64 dims this is what v hv used in training so wee need to use exactly the same dims
# here also

test_image


test_image = image.img_to_array(test_image)
# Also in our first layer below it is a 3D array
# Step 1 - Convolution
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# this will convert from a 3D img to 3D array
test_image # shld gv us (64,64,3)


test_image = np.expand_dims(test_image, axis = 0)
# axis specifies the position of indx of the dimnsn v r addng
# v need to add the dim in the first position
test_image # now it shld show (1,64,64,3)

x=preprocess_input(test_image)
x

#result = classifier.predict(test_image)
result = classifier.predict(x)
# v r trying to predict
result # gv us 1


print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

test_image = image.load_img('/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/cnn_imgs_dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image
test_image = image.img_to_array(test_image)
test_image.shape
test_image = np.expand_dims(test_image, axis = 0)
test_image.shape
y=preprocess_input(test_image)
y
#result = classifier.predict(test_image)
result = classifier.predict(y)
result
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)














