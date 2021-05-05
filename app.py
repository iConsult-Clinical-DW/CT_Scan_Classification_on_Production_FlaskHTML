from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from flask import Flask, request, jsonify, render_template
import argparse

app = Flask(__name__)

model = load_model('model_deep_cnn.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('public/index.html')

@app.route('/',methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        img = uploaded_file.filename
        img = image.load_img(img, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        prediction = model.predict(img_preprocessed)
        if prediction == [[1.]]:
            output = "CANCER"
        else:
            output = "NORMAL"
        args = True
        
    return render_template('public/index.html', args=args, prediction_text='The CT scan image is predicted to be of class {}'.format(output))

if __name__ == "__main__":
    app.run()