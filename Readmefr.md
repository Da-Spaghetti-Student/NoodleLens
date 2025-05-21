# Noodlens, Mon application de reconnaissance faciale

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/Da-Spaghetti-Student/NoodleLens/blob/main/README.md)
![py](https://img.shields.io/badge/python-3.11.2-3C79AB)

## Description
Mon projet est une simple application web tournant sur Flask qui permet aux utilisateurs de télécharger une image. Il utilisera ensuite un modèle CNN pour reconnaître un visage et une émotion humaine dans l’image. 

Il comprend actuellement 7 émotions :
- Fâché
- Dégoûté
- Effrayé
- Heureux
- Triste
- Surpris
- Neutre

Il y a aussi un fichier réservé pour l'entrainement du modèle de convolution, ainsi que deux scripts en python pour de la reconnaissance en temps réel et à l'aide d'une image. 

## Installation
1. Cloner le repository:
   ```bash
   git clone https://github.com/Da-Spaghetti-Student/NoodleLens.git
   
2.Fichiers requis dans la racine de l'application.

Le fichier "haarcascade_frontalface_default.xml" provenant de opencv et votre fichier "model.h5" doivent être present dans la racine de l'application.
