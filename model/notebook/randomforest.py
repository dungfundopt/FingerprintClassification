from sklearn.model_selection import train_test_split, learning_curve
import os
import cv2
import sys
sys.path.insert(0, 'C:\\Users\\ADMIN\\Fingerprint_Recognition_GROUP8\\model')
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from tqdm import tqdm
from feature_extract.feature_extraction import final_bow_feature, image_preprocessing, extract_sift, bow_vector
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
X_features, Y = final_bow_feature()
X_train = []
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size = 0.2, random_state=42)
print(len(X_train), " ", len(X_test))