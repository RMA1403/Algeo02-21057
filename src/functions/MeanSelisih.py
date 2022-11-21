import cv2
import numpy as np
from glob import glob
import time

import Covarian as cov
import Eigenvalue as eigval
import EucDistance as eucdis
import mean_selisih as Mean

# path = "./testImage"
# test_image = glob(path + "/*.png") + glob(path + "/*.jpg")#
# imgcount = len(test_image)
# imglist = []
# for i in range(len(test_image)):
#     img = cv2.resize(cv2.cvtColor(cv2.imread(test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
#     img = img.reshape(len(img)**2,1)
#     imglist.append(img)

# print(np.asarray(imglist))
# imgarr = np.asarray(imglist)

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

    matA = array[0]-m
    # selisih = []
    
    for i in range(1, len(array)):
        matA = np.concatenate((matA,(array[i]-m)), axis = 1)
        # selisih.append(array[i]-m)

    return matA, m
<<<<<<< HEAD:src/mean_selisih.py

#print(mean_selisih(imgarr)[1])
=======

# imgarr = [[[1],[2],[3]],[[1],[2],[3]]]

# print(mean_selisih(imgarr))
>>>>>>> testing:src/functions/MeanSelisih.py
