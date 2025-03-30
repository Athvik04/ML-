# # Gestione degli imports:
# import os
# from skimage import filters
# from skimage import io
# from skimage.io import imread
# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
# from skimage.feature import hog
# from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import cross_val_predict
# from sklearn.preprocessing import StandardScaler, Normalizer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn import svm

# import cv2 as cv

# # Per dataset pickle:
# import joblib

# # Varie ed eventuali
# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# resize_x = 64
# resize_y = 128

# #-----------------------------------------------------------------------------------------------------------------------#
# # VARIABILI BOOLEANE DEL PROGRAMMA:
# prepare_data = False
# prepare_dataset = True
# print_dataset_info = True

# #-----------------------------------------------------------------------------------------------------------------------#
# # TRASFORMATORE PER ESTRARRE GLI HISTOGRAM OF ORIENTED GRADIENTS:

# class HogTransformer(BaseEstimator, TransformerMixin):
#     """
#     Expects an array of 2d arrays (1 channel images)
#     Calculates hog features for each img
#     """
#     def __init__(self, y=None, orientations=9,
#             pixels_per_cell=(8, 8),
#             cells_per_block=(3, 3), block_norm='L2-Hys'):
#         self.y = y
#         self.orientations = orientations
#         self.pixels_per_cell = pixels_per_cell
#         self.cells_per_block = cells_per_block
#         self.block_norm = block_norm

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):

#         def local_hog(X):
#             return hog(X,
#                 orientations=self.orientations,
#                 pixels_per_cell=self.pixels_per_cell,
#                 cells_per_block=self.cells_per_block,
#                 block_norm=self.block_norm)
        
#         try: # parallel
#             return np.array([local_hog(img) for img in X])
#         except:
#             return np.array([local_hog(img) for img in X])

# #-----------------------------------------------------------------------------------------------------------------------#
# # ACQUISISCO LE IMMAGINI DALLA WEBCAM CON OPENCV, LE SALVO IN CARTELLE APPOSITE:

# def equalize_imgs(img):
#     R, G, B = cv.split(img)
#     output1_R = cv.equalizeHist(R)
#     output1_G = cv.equalizeHist(G)
#     output1_B = cv.equalizeHist(B)
#     equ = cv.merge((output1_R, output1_G, output1_B))
#     return equ

# if prepare_data:
#     print("Connecting to camera...")
#     camera = cv.VideoCapture(0)
#     print("Connection estabilished")

#     keyboard = input("Insert text: ")
#     match keyboard:
#         case '1':
#             for i in range(200):
#                 ret, frame = camera.read()
#                 equ = equalize_imgs(frame)
#                 cv.imwrite("./class_1/class_1_{}.jpg".format(i), equ)
#         case '2':
#             for i in range(200):
#                 ret, frame = camera.read()
#                 equ = equalize_imgs(frame)
#                 cv.imwrite("./class_2/class_2_{}.jpg".format(i), equ)

#         case _:
#             print("Case out of bounds")

# #-----------------------------------------------------------------------------------------------------------------------#
# # PREPARO IL MIO DATASET IN FORMATO .PKL:

# if prepare_dataset:
#     path = "./"

#     files = os.listdir(path)

#     classes = {}

#     for element in files:
#         if (not element.endswith(".py")):
#             img_path = path + element
#             images = os.listdir(img_path)
#             for image_number in range(len(images)):
#                 images[image_number] = img_path + "/" + images[image_number]
#             classes[element] = images

#     # Ora ho un dizionario così composto:
#     # 'class_1' : ['./class_1/image0.jpg', './class_1/image1.jpg', ...]

#     # Preparo il dataset applicando trasformazioni alle immagini:
#     dataset = dict()
#     dataset['description'] = 'My dataset composed of two classes, namely C1 and C2'
#     dataset['label'] = []
#     dataset['filename'] = []
#     dataset['data'] = []
#     pklname = f"" + str(resize_y) + "px" + str(resize_x) + "px.pkl"

#     for elements in classes:
#         for image_number in classes[elements]:
#             work_on_image = io.imread(image_number)
#             work_on_image = resize(work_on_image, (resize_y, resize_x), anti_aliasing=True)
#             work_on_image = color.rgb2gray(work_on_image)
#             print("class: {} - image number: {}".format(elements, image_number))
#             dataset['label'].append(elements)
#             dataset['filename'].append(image_number)
#             dataset['data'].append(work_on_image)

#     joblib.dump(dataset, pklname)

# #-----------------------------------------------------------------------------------------------------------------------#
# # ADDESTRO IL CLASSIFICATORE, FACENDO LOADING DEI DATI:

# # data = joblib.load(f'{}px{}px.pkl'.format(resize_y, resize_x))
# data = joblib.load(f"" + str(resize_y) + "px" + str(resize_x) + "px.pkl")

# if print_dataset_info:
#     print('number of samples: ', len(data['data']))
#     print('keys: ', list(data.keys()))
#     print('description: ', data['description'])
#     print('image shape: ', data['data'][0].shape)
#     print('labels:', np.unique(data['label']))
#     Counter(data['label'])

# labels = np.unique(data['label'])

# X = np.array(data['data'])
# y = np.array(data['label'])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, 
#     y, 
#     test_size=0.2, 
#     shuffle=True,
#     random_state=42,
# )

# hogify = HogTransformer(
#     pixels_per_cell = (8, 8), 
#     cells_per_block = (2, 2), 
#     orientations = 9, 
#     block_norm = 'L2-Hys'
# )

# scalify = StandardScaler()

# # call fit_transform on each transform converting X_train step by step
# X_train_hog = hogify.fit_transform(X_train)
# X_train_prepared = scalify.fit_transform(X_train_hog)

# print(X_train_prepared.shape)

# sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
# sgd_clf.fit(X_train_prepared, y_train)

# # Effettuo le trasformazioni sui sample di test:
# X_test_hog = hogify.transform(X_test)
# X_test_prepared = scalify.transform(X_test_hog)

# #-----------------------------------------------------------------------------------------------------------------------#
# # EFFETTUO I TEST:

# y_pred = sgd_clf.predict(X_test_prepared)
# print(np.array(y_pred == y_test)[:10])
# print('')
# print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

# #-----------------------------------------------------------------------------------------------------------------------#

# # save the model to disk
# joblib.dump(sgd_clf, 'sgd_model.pkl')

# print("Model saved on disk.")


# Importing necessary libraries
import os
from skimage import filters, io, color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np
import joblib
import cv2 as cv
import matplotlib.pyplot as plt

# Image resizing dimensions
resize_x = 64
resize_y = 128

# Boolean variables for program settings
prepare_data = False
prepare_dataset = True
print_dataset_info = True

# HOG Transformer class for extracting Histogram of Oriented Gradients
class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y=None, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block, block_norm=self.block_norm)
        
        return np.array([local_hog(img) for img in X])

# Function to equalize images
def equalize_imgs(img):
    R, G, B = cv.split(img)
    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    equ = cv.merge((output1_R, output1_G, output1_B))
    return equ

# Capture images from webcam and save them to folders
if prepare_data:
    print("Connecting to camera...")
    camera = cv.VideoCapture(0)
    print("Connection established")

    keyboard = input("Insert text: ")
    match keyboard:
        case '1':
            for i in range(200):
                ret, frame = camera.read()
                equ = equalize_imgs(frame)
                cv.imwrite(f"./class_1/class_1_{i}.jpg", equ)
        case '2':
            for i in range(200):
                ret, frame = camera.read()
                equ = equalize_imgs(frame)
                cv.imwrite(f"./class_2/class_2_{i}.jpg", equ)
        case _:
            print("Case out of bounds")

# Prepare the dataset and save as a .pkl file
if prepare_dataset:
    path = "./"
    files = os.listdir(path)
    classes = {}

    for element in files:
        img_path = os.path.join(path, element)
        if os.path.isdir(img_path):  # Check if path is a directory
            images = os.listdir(img_path)
            for image_number in range(len(images)):
                images[image_number] = os.path.join(img_path, images[image_number])
            classes[element] = images
        else:
            print(f"Skipping non-directory file: {element}")

    # Create the dataset dictionary
    dataset = {
        'description': 'My dataset composed of two classes, namely C1 and C2',
        'label': [],
        'filename': [],
        'data': []
    }
    pklname = f"{resize_y}px{resize_x}px.pkl"

    for elements in classes:
        for image_number in classes[elements]:
            work_on_image = io.imread(image_number)
            work_on_image = resize(work_on_image, (resize_y, resize_x), anti_aliasing=True)
            work_on_image = color.rgb2gray(work_on_image)
            print(f"class: {elements} - image number: {image_number}")
            dataset['label'].append(elements)
            dataset['filename'].append(image_number)
            dataset['data'].append(work_on_image)

    # Save the dataset to disk
    joblib.dump(dataset, pklname)

# Load the dataset for training
data = joblib.load(f"{resize_y}px{resize_x}px.pkl")

if print_dataset_info:
    print('Number of samples:', len(data['data']))
    print('Keys:', list(data.keys()))
    print('Description:', data['description'])
    print('Image shape:', data['data'][0].shape)
    print('Labels:', np.unique(data['label']))
    Counter(data['label'])

labels = np.unique(data['label'])
X = np.array(data['data'])
y = np.array(data['label'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Set up the pipeline for HOG features and scaling
hogify = HogTransformer(pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
scalify = StandardScaler()

# Transform the training data
X_train_hog = hogify.fit_transform(X_train)
X_train_prepared = scalify.fit_transform(X_train_hog)
print(X_train_prepared.shape)

# Train an SGD classifier
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)

# Transform the test data
X_test_hog = hogify.transform(X_test)
X_test_prepared = scalify.transform(X_test_hog)

# Evaluate the classifier
y_pred = sgd_clf.predict(X_test_prepared)
print(np.array(y_pred == y_test)[:10])
print('')
print('Percentage correct:', 100 * np.sum(y_pred == y_test) / len(y_test))

# Save the trained model
joblib.dump(sgd_clf, 'sgd_model.pkl')
print("Model saved on disk.")
