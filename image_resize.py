import glob
from scipy.io import loadmat
import numpy as np
import cv2

files = glob.glob("C:/Users/Vinuthna/Desktop/Plant Species/Chinar (P11)/healthy/*")
files = np.sort(files)
    
for i in range(len(files)):
    print(files[i], i)
    img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
    dim = (720, 480)
    resized = cv2.resize(img, (720, 480), interpolation = cv2.INTER_AREA)
    cv2.imwrite("C:/Users/Vinuthna/Desktop/resize/Chinar/healthy/" + "chinar_healthy_" + str(i+1) + ".jpg",resized)