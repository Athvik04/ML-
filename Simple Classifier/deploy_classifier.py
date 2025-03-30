# Standard imports:
import joblib
from skimage import io
from skimage.io import imread
from skimage import color
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale, resize
from skimage.feature import hog
import numpy as np

# Visualizzazione
from skimage import data, exposure
import matplotlib.pyplot as plt
import cv2

def equalize_imgs(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    equ = cv2.merge((output1_R, output1_G, output1_B))
    return equ

# Parametri per ottenere l'HOG: devono essere gli stessi della classe HOG_TRANSFORMER (vedi file classify.py)
resize_x = 64
resize_y = 128

orientations=9
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
block_norm='L2-Hys'

#------------------------------------------------------------------------------------------------------------#
# Importo il classificatore:
custom_classifier = joblib.load(f'sgd_model.pkl')

camera = cv2.VideoCapture(0)

while(True):
    ret, frame = camera.read()
    image_to_classify = equalize_imgs(frame)

    # Eseguo trasformazioni:
    resized_image = resize(image_to_classify, (resize_y, resize_x), anti_aliasing=True)
    grey_image = color.rgb2gray(resized_image)
    X_test_hog, hog_image = hog(grey_image, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, block_norm = block_norm, visualize=True)
    
    # Se volessi visualizzare:
    # # plt.imsave("visualize.jpg", hog_image, cmap="gray")

    # Per un singolo campione, mi tocca effettuare il reshape dell'array:
    # Così la shape risulta: (1, feature_vector_len (ad esempio:1764))
    X_test_hog = X_test_hog.reshape(1, -1)
    y_pred = custom_classifier.predict(X_test_hog)
    print(y_pred[0])

print("Fine programma.")
