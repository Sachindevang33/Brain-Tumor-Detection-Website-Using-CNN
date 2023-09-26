import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import cv2
import imutils
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt



app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

def crop_brain_contour(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
    
    return new_image, gray, thresh, c

@app.route("/", methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join("D:/DIP Project/static/assets/images/", filename))
        return redirect(url_for('predict', filename=filename))
    return render_template("index.html")

@app.route("/predict/<filename>", methods = ['GET', 'POST'])
def predict(filename): 
    
    image = cv2.imread(os.path.join("D:/DIP Project/static/assets/images/", filename))
    new_image,gray,thresh,c = crop_brain_contour(image, plot=False)
     
    image_width, image_height = (240,240)
    new_image = cv2.resize(new_image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("D:/DIP Project/static/assets/images/after_contour.jpg", new_image) 
    new_image = new_image/255
    obj = []
    obj.append(new_image)
    obj = np.array(obj)
    result  = model.predict(obj)
    source = os.path.join("assets/images/", filename)
    if(result > 0.5):
        res = "Given Image contain Tumour"
    else:
        res = "Given Image doesnt contain Tumour"

    result = list(result)
    result = result[0]
    result = result[0]*100
    #resa = result.astype(float)
    #result = resa
    result = "{:.2f}".format(result)
    cv2.imwrite("D:/DIP Project/static/assets/images/gray.jpg", gray)    
    cv2.imwrite("D:/DIP Project/static/assets/images/thresh.jpg", thresh)  
        
    return render_template("predict.html", result = result, source = source, res = res,filename = filename, gray = gray)



if __name__ == "__main__":
    port = int(5000)
    app.static_folder = 'static'
    app.run(host='0.0.0.0', port=port, debug=True)
    
