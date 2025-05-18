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
        
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) ))
        
    val = finds()
    imgpred = os.path.join(app.config['UPLOAD_FOLDER_LINK'])+f.filename
    return render_template('pred.html', ss = val, pimage = imgpred)
    
def allowed_file(filename):
    # Check if the filename contains a dot
    if '.' not in filename:
        return False
    # Split the filename and get the extension
    extension = filename.rsplit('.', 1)[1]
    # Convert the extension to lowercase
    extension = extension.lower()
    # Check if the extension is in the allowed extensions
    return extension in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def finds():
    model = models.load_model("model.h5", compile = False)

    vals = {0: "Colere",1: "Degout",2: "Peur",3: "Bonheur",4: "Triste",5: "Surpris",6: "Neutre",}
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    f= request.files['file']
    
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'])+"\\"+f.filename)
    if img is None:
        print("Erreur: Impossible de lire l'image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 140, 0), 2)
        face = gray[y : y + h, x : x + w]
        face = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # prédire l'émotion
        emotion_prediction = model(face)
        maxindex = int(np.argmax(emotion_prediction))

        
        cv2.putText(
            gray, vals[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 236, 0), 2
        )
        
    pred = vals[maxindex]
    print(pred)
    return str(vals[maxindex])

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
   app.run()