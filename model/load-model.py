print("""
 _______                    .___.__         .____                                
 \      \   ____   ____   __| _/|  |   ____ |    |    ____   ____   ______       
 /   |   \ /  _ \ /  _ \ / __ | |  | _/ __ \|    |  _/ __ \ /    \ /  ___/       
/    |    (  <_> |  <_> ) /_/ | |  |_\  ___/|    |__\  ___/|   |  \\___ \        
\____|__  /\____/ \____/\____ | |____/\___  >_______ \___  >___|  /____  >       
        \/                   \/           \/        \/   \/     \/     \/        
                                                                                 
                                                                                 
  ______   ______   ______   ______   ______   ______   ______   ______   ______ 
 /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/  /_____/ 
                                                                                 
      
Loading...
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
    0: "Fâché",
    1: "Dégoûté",
    2: "Effrayé",
    3: "Heureux",
    4: "Triste",
    5: "Surpris",
    6: "Neutre",
}

# Open webcam
video = cv2.VideoCapture(0)

while True:
    # Read the video input frames by frames
    ret, img = video.read()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle on face(s)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 0), 2)
        face = gray[y : y + h, x : x + w]
        face = numpy.expand_dims(numpy.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # Predict emotions
        emotion_prediction = model(face)
        maxindex = int(numpy.argmax(emotion_prediction))
        cv2.putText(
            img, emotions[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 236, 0), 2
        )

    cv2.imshow("NoodleLens", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
