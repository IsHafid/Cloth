from flask import Flask, render_template, request
import os
from keras.models import load_model
import keras.utils as image
import numpy as np


UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__)
model = load_model('.\model\ismailHAF.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path,grayscale=True, target_size=(28, 28))
    i = image.img_to_array(i)
    i = i.reshape(1, 28, 28,1)
    i=i.astype('float32')
    #i=i/ 255.0
    i=255 - i
    prediction_test = model.predict(i)
    print(prediction_test)

    return class_names[np.argmax(prediction_test[0])]


@app.route('/')
def index():
    return (render_template('index.html'))


@app.route('/upload', methods=['POST'])
def upload_file():
    img = request.files['file']
    img.save(os.path.join(UPLOAD_FOLDER, img.filename))

    img_path = "static/uploads/" + img.filename
    p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)
    #return 'file uploaded successfully'


app.run()