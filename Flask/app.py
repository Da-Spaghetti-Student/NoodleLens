import os

import numpy as np
import cv2

from keras import models
from flask import Flask, render_template, send_from_directory, request
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')

app.config['UPLOAD_FOLDER'] = 'static\\uploaded\\images'
app.config['UPLOAD_FOLDER_LINK'] = 'uploaded/images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    if request.method == 'POST':
        f = request.files['file']    
        if f.filename == '':
            return "No selected file"

   #Upload file
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) ))
        
    val = finds()
   #Path for the image
    imgpred = os.path.join(app.config['UPLOAD_FOLDER_LINK'])+f.filename
    return render_template('pred.html', ss = val, pimage = imgpred)

# Check if the extension is in the allowed extensions
def allowed_file(filename):
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1]
    extension = extension.lower()
    return extension in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def finds():
   #Import CNN model
    model = models.load_model("model.h5", compile = False)

   #The model's emotions
    vals = {0: "Colere",1: "Degout",2: "Peur",3: "Bonheur",4: "Triste",5: "Surpris",6: "Neutre",}
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    f= request.files['file']

   #Read the image with opencv
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'])+"\\"+f.filename)
    if img is None:
        print("Erreur: Its impossible to read the image.")
        return

   #Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 140, 0), 2)
        face = gray[y : y + h, x : x + w]
        face = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # Emotion prediction
        emotion_prediction = model(face)
        maxindex = int(np.argmax(emotion_prediction))

   #Get the dominant emotion
    pred = vals[maxindex]
    print(pred)
    return str(vals[maxindex])

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
   app.run()
