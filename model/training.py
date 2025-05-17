import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras.layers
import keras.optimizers
import numpy, pandas, keras, cv2
from sklearn.model_selection import train_test_split

# Recuperation des images(dataset)
file_path = os.path.dirname(os.path.realpath(__file__)) + "\\fer2013.csv"
data = pandas.read_csv(file_path)
pixels = data["pixels"].tolist()

# Initialisation de la largeur et de la hauteur des images
width, height = 48, 48
faces = []

# redimensionner les images
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(" ")]
    face = numpy.asarray(face).reshape(width, height)
    face = cv2.resize(face.astype("uint8"), (width, height))
    faces.append(face.astype("float32"))

# convertir la liste en tableau numpy
faces = numpy.asarray(faces)
faces = numpy.expand_dims(faces, -1)
emotions = pandas.get_dummies(data["emotion"]).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    faces, emotions, test_size=0.2, shuffle=True
)

model = keras.Sequential()
# Exclusif au CNN:
#   Conv2D -> Utilise les kernels
#   MaxPooling2D -> Simplifie les valeurs récupérer par les kernels
#   Dropout -> Reduit l'overfitting

# activation Layer:
#   relu -> ne permet pas d'avoir des nombres négatifs
#   softmax -> determine le nombre de couches

# Présent dans tout les ANN:
#   Dense -> Connecte les nodes(neurones) entre eux
#   Flatten -> Transforme les valeurs récupérée par pooling en 1 dimension car, + facile à manipuler


model.add(
    keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", input_shape=(X_train.shape[1:])
    )
)
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(7, activation="softmax"))

# Compilation du modèle
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Nadam(learning_rate=0.0001, decay=1e-7),
    metrics=["accuracy"],
)
model.fit(
    numpy.array(X_train),
    numpy.array(y_train),
    epochs=50,
    verbose=1,
    validation_data=(numpy.array(X_test), numpy.array(y_test)),
    shuffle=True,
)

# Sauvegarde du modèle
model.save("model.h5")
