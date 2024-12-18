import cv2
#import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
#import sklearn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import normalize
#sys.path.append('../preprocess')
sys.path.insert(0, 'C:\\Users\\ADMIN\\FingerprintClassification\\model')
from preprocess.preprocessing3 import preprocessing_normal
def read_data(label2num):
    id_anh = 0
    X=[]
    Y=[]
    #output_folder = "C:\\Users\\ADMIN\\FingerprintClassification\\model\\preprocessed_fingerprint"
    for label in os.listdir(os.path.join("preprocessed_fingerprint")):
        id_anh+=1
        img = cv2.imread(os.path.join('preprocessed_fingerprint',str(id_anh)+'.png'))
        X.append(img)
        Y.append(label2num["fingerprint"])
        if id_anh>=4000: break
    for label in os.listdir(os.path.join("preprocessed_fingerprint")):
        id_anh+=1
        img = cv2.imread(os.path.join('preprocessed_fingerprint',str(id_anh)+'.png'))
        X.append(img)
        Y.append(label2num["fingerprint_noise"])
    return X, Y

def image_preprocessing(X):
    img_prep = []
    for i in range(len(X)):
        abc = preprocessing_normal(X[i])
        img_prep.append(abc)
    return img_prep

def extract_sift(X):
    image_descriptors = []
    #sift = cv2.SIFT_create()
    sift = cv2.SIFT_create()
    for i in range(len(X)):
        _, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)
    return image_descriptors

#print(len(image_descriptors))
'''for i in range(3):
    print('anh {} gom {} features'.format(i, len(image_descriptors[i])))'''
def pca_sift(image_descriptors):
    image_descriptors_pca = []
    for des in image_descriptors:
        pca = PCA(n_components=32)

    # Fit và transform dữ liệu
        pca.fit(des)
        des_pca = pca.transform(des)
        image_descriptors_pca.append(des_pca)
    return image_descriptors_pca
#len(all_descriptors)
def kmeans_bow(all_descriptors, numberofclus):
    bow_dict = []
    kmeans = KMeans(n_clusters=numberofclus)
    kmeans.fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_

    return bow_dict

#print(len(BoW))
def bow_vector(image_descriptors, BoW, num_cluster):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0]*num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis = 1)
            for j in argmin:
                features[j]+=1
        X_features.append(features)
    X_features = normalize(X_features, norm='l2', axis=1)
    return X_features
def final_bow_feature():
    label2num = {'fingerprint_noise':0, 'fingerprint':1}
    X, Y = read_data(label2num)

    processed_img = image_preprocessing(X)

    image_descriptors = extract_sift(processed_img)

    image_descriptors_pca = pca_sift(image_descriptors)

    all_descriptors = [dess for des in image_descriptors_pca if des is not None for dess in des]

    num_cluster=50
    if not os.path.isfile('ml_data/bow_vector_cluster_50.pkl'):
        BoW = kmeans_bow(all_descriptors, num_cluster)
        pickle.dump(BoW, open('ml_data/bow_vector_cluster_50.pkl', 'wb'))
    else:
        BoW = pickle.load(open('ml_data/bow_vector_cluster_50.pkl', 'rb'))
    X_features = bow_vector(image_descriptors_pca, BoW, num_cluster)
    return X_features, Y