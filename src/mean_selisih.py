import cv2
import numpy as np
from glob import glob
import time
import os

import Covarian as cov
import Eigenvalue as eigval
import EucDistance as eucdis
import mean as Mean


#path = "./testImage"
#test_image = glob(path + "/*.png") + glob(path + "/*.jpg")
#imgcount = len(test_image)
#print(imgcount)
#imglist = []
#for i in range(len(test_image)):
    #img = cv2.resize(cv2.cvtColor(cv2.imread(
            #test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
    #img = img.reshape(len(img)**2,1)
    #imglist.append(img)

#print(np.asarray(imglist))
#imgarr = np.asarray(imglist)
#print(len(imgarr))

def mean_selisih(array):
    row = len(array[0])  # baca baris dari matrix pertama
    col = len(array[0][0])
    m = [[0 for i in range(col)]for j in range(row)]
    count = 0

    while count < len(array):
        m += array[count] # menambahkan semua matrix
        count += 1

    for i in range(row):
        for j in range(col):
            # mengalikan hasil akhir matrix dengan 1/M
            m[i][j] = 1/count * m[i][j]

    matA = [[]for i in range(len(imgarr[0]))]
    
    for i in range(len(imgarr)):
        matA = np.concatenate((matA,(imgarr[i]-m)), axis = 1)
    
    return matA
    



#print(mean(imgarr))


    