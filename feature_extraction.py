import cv2
import glob
import numpy as np
import mahotas
import pandas as pd
import math

files_1 = glob.glob("Plant Species/diseased/*")
files_2 = glob.glob("Plant Species/healthy/*")

def hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    for i in range(0,7):
        feature[i] = -1* math.copysign(1.0, feature[i]) * math.log10(abs(feature[i]))
    cv2.normalize(feature, feature)
    return feature
    
    
def haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 
 
def histogram(image, mask=None):
    color = ('b','g','r')
    hist = []
    for i,col in enumerate(color):
        temp = cv2.calcHist([image],[i],None,[16],[0,256])
        cv2.normalize(temp, temp)
        for j in range(len(temp)) :
            hist.append(temp[j]);
    hist = np.array(hist)
    hist = hist.flatten()
    return hist

diseased_quantity = [254, 232, 120, 142, 345, 124, 77, 265, 272, 276]
healthy_quantity = [179, 220, 103, 277, 279, 133, 159, 170, 287, 322]
    
global global_features

count = 1
total_count = sum(diseased_quantity) + sum(healthy_quantity)

for i in range(len(files_1)):
    print(files_1[i], count, "/", total_count)
    image = cv2.imread(files_1[i], cv2.IMREAD_UNCHANGED)
    if i :
        global_features = np.vstack([global_features, np.hstack([histogram(image),haralick(image), hu_moments(image)])])
        count = count+1
    else :
        global_features = np.hstack([histogram(image),haralick(image), hu_moments(image)])
        count = count+1


for i in range(len(files_2)):
    print(files_2[i], count, "/", total_count)
    image = cv2.imread(files_2[i], cv2.IMREAD_UNCHANGED)
    global_features = np.vstack([global_features, np.hstack([histogram(image),haralick(image), hu_moments(image)])])
    count = count+1


diseased_classes=[]
for i in range(len(diseased_quantity)) :
    vec = (i+1)*np.ones((diseased_quantity[i],1))
    for j in range(len(vec)) :
        diseased_classes.append(vec[j])
    
diseased_classes = np.array(diseased_classes)
diseased_classes = np.reshape(diseased_classes, (sum(diseased_quantity), 1))

healthy_classes=[]
for i in range(len(healthy_quantity)) :
    vec = (i+1)*np.ones((healthy_quantity[i],1))
    for j in range(len(vec)) :
        healthy_classes.append(vec[j])
    
healthy_classes = np.array(healthy_classes)
healthy_classes = np.reshape(healthy_classes, (sum(healthy_quantity), 1))


target_classes = np.vstack([diseased_classes, healthy_classes])
print(target_classes.shape)

target_condition = np.vstack([np.ones((sum(diseased_quantity), 1)), 2*np.ones((sum(healthy_quantity), 1))])
print(target_condition.shape)


global_features = np.hstack([global_features, target_classes, target_condition])

data_frame = pd.DataFrame(global_features)
data_frame.to_csv("plant_species_features.csv")

print("<-----------------------------------------Feature Extraction Completed----------------------------------------->")