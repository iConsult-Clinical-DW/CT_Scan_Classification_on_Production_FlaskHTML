# -*- coding: utf-8 -*-
## https://www.kaggle.com/mohamedhanyyy/chest-ctscan-images

### IMPORTING The REQUIRED LIBRARIES
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import matplotlib.pyplot as plt
import pickle

training_datagenerator = ImageDataGenerator(rescale=1/255,zoom_range=0.2,rotation_range=15) ### To generate Training Data
validation_datagenerator = ImageDataGenerator(rescale=1/255) ### To generate Validation Data

training_data = training_datagenerator.flow_from_directory(
        directory='D:/SYR ADS/iConsult/CT_Scan_Classification_on_Production_FlaskHTML/Lung_CT_scans_cancerDATA/train',
        class_mode='binary', ## Since this is a binary classification problem, we will use binary_crossentropy loss
        target_size=(224,224),  ## To resize our images 256x256 pixels.  
        batch_size=29
)
print("Training data created")

validation_data = validation_datagenerator.flow_from_directory(
        directory='D:/SYR ADS/iConsult/CT_Scan_Classification_on_Production_FlaskHTML/Lung_CT_scans_cancerDATA/test',
        class_mode='binary', ## Since this is a binary classification problem, we will use binary_crossentropy loss
        target_size=(224,224),  ## To resize our images 256x256 pixels.  
        batch_size=21
)

model2_deep=Sequential()
model2_deep.add(Conv2D(32,(3,3),input_shape=(224,224,3))) ### 1st Convolutional Hidden layer with 32 filters of size 3x3 pixels
model2_deep.add(MaxPooling2D(pool_size=(2, 2)))  ### 1st MaxPooling layer of size 2x2
model2_deep.add(Conv2D(32,(3,3)))   
model2_deep.add(MaxPooling2D(pool_size=(2, 2)))
model2_deep.add(Conv2D(32,(3,3)))
model2_deep.add(MaxPooling2D(pool_size=(2, 2)))
model2_deep.add(Conv2D(32,(3,3)))
model2_deep.add(MaxPooling2D(pool_size=(2, 2)))
model2_deep.add(Conv2D(32,(3,3)))
model2_deep.add(MaxPooling2D(pool_size=(2, 2)))
model2_deep.add(Flatten())      ### Flattening the 2D data to 1D to be used for the Dense layers
model2_deep.add(Dense(512,activation='relu'))  ### 1st Dense Layer with Relu Activation Function
model2_deep.add(Dropout(0.2))
model2_deep.add(Dense(512,activation='relu'))
model2_deep.add(Dense(1,activation='sigmoid'))  ### Output Layer with Sigmoid Activation Function

#### Using 20 epochs again
epoch=20
#### Using the binary_crossentropy loss function again
model2_deep.compile(optimizer='rmsprop',metrics='accuracy',loss='binary_crossentropy')
### Using the fit_generator() function to Train the CNN on the Train Data & evaluate Training & Validation Accuracy and Loss
hist_deep = model2_deep.fit_generator(training_data,
                        steps_per_epoch=12,
                        validation_data=validation_data,
                        validation_steps=6,
                        epochs=epoch)
deep_cnn_df = hist_deep.history
plt.plot(range(1,epoch+1),deep_cnn_df['loss'],label='training loss')
plt.plot(range(1,epoch+1),deep_cnn_df['val_loss'],label='validation loss')
plt.legend()
plt.title('Training & Validation Losses after every Epoch')
plt.xlabel('Epochs')
plt.ylabel('LOSS')
plt.show()

#Pkl_Filename = "model.pkl"
#with open(Pkl_Filename, 'wb') as file:  
#    pickle.dump(model2_deep, file)

model2_deep.save('model_deep_cnn.h5') 
   
print("Model FILE CREATED")