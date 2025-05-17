print("""
 _______                    .___.__         .____                                
 \      \   ____   ____   __| _/|  |   ____ |    |    ____   ____   ______       
 /   |   \ /  _ \ /  _ \ / __ | |  | _/ __ \|    |  _/ __ \ /    \ /  ___/       
/    |    (  <_> |  <_> ) /_/ | |  |_\  ___/|    |__\  ___/|   |  \\___ \        
\____|__  /\____/ \____/\____ | |____/\___  >_______ \___  >___|  /____  >       
        \/                   \/           \/        \/   \/     \/     \/        
                                                                                 
                                                                                 
  ______   ______   ______   ______   ______   ______   ______   ______   ______ 
 /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/ 
                                                                                 
      
Chargement du modèle...
""")

import warnings
warnings.simplefilter("ignore", UserWarning)

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2, numpy
from keras import models

model = models.load_model("model.h5", compile = False)

emotions = {
    0: "Colere",
    1: "Degout",
    2: "Peur",
    3: "Bonheur",
    4: "Triste",
    5: "Surpris",
    6: "Neutre",
}

def detect_faces(image_path):
    #Pour éviter les erreurs de paths.
    image_path = os.path.dirname(os.path.realpath(__file__)) +"\\"+image_path
    
    # Importation du classeur de detection de visage d'opencv.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img = cv2.imread(image_path)
    if img is None:
        print("Erreur: Impossible de lire l'image.")
        return

    # convertir en échelle de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # détecter le visage
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # dessiner un rectangle sur le visage
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 0), 2)
        face = gray[y : y + h, x : x + w]
        face = numpy.expand_dims(numpy.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # prédire l'émotion
        emotion_prediction = model(face)
        maxindex = int(numpy.argmax(emotion_prediction))
        cv2.putText(
            img, emotions[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 236, 0), 2
        )
    # Affichage de l'image.
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Le nom et l'extention du fichier.
detect_faces("Obama.jpg")
