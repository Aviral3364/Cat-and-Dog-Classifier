import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

####################################################################################

train_datagen = ImageDataGenerator(rescale=1./255,
								   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
	                                              batch_size=32,
                                                  target_size=(64, 64),
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')


#########################################################################################
#Designing the Model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform', 
	                    input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(128, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(training_set, epochs = 30)
preds = model.evaluate(test_set)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

test_image = image.load_img('dataset/single_prediction/test.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
prediction = ''
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'

print('The predicted result is:',prediction)

model.summary()
model.save('Cat_and_Dog_Classifier', include_optimizer = True)

print("\n")
print("***********************************************************************************************************")
print("THANK YOU FOR AVAILING THIS SERVICE")
print("This CNN had been implemented by AVIRAL SINGH")
print("***********************************************************************************************************")
