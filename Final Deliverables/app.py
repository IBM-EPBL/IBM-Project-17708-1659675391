import requests
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)
model=load_model('ECG.h5')


@app.route('/')
@app.route('/home.html')
def home():
    return render_template('home.html')


@app.route('/about-us.html')
def about():
    return render_template('about-us.html')


@app.route('/classify.html', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        f = request.files['file']  # requesting the file
        basepath = os.path.dirname('__file__')  # storing the file directory
        filepath = os.path.join(basepath, "uploads", f.filename)  # storing the file in uploads folder
        f.save(filepath)  # saving the file

        img = image.load_img(filepath, target_size=(64, 64))  # load and reshaping the image
        x = image.img_to_array(img)  # converting image to an array
        x = np.expand_dims(x, axis=0)  # changing the dimensions of the image

        pred = np.argmax(model.predict(x), axis=1)
        print("prediction", pred)  # printing the prediction
        index=['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction', 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']


        result = str(index[pred[0]])

        x = result
        print(x)

        return render_template('result.html', result=x)

    else:
        return render_template('classify.html')



if __name__ == '__main__':
    app.run()