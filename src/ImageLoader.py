import cv2
import numpy as np
from glob import glob
import matplotlib as plt

import Covarian as cov
import Eigenvalue as eigval
import Eigenvector as eigvec
import EucDistance as eucdis
import mean

test_image = glob("./test/testImage/*.png")

imgcount = len(test_image)
img = cv2.imread(test_image[0])
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row = img1.shape[0]
col = img1.shape[1]

mattotal = [[0 for i in range(col)]for j in range(row)]

for i in range(imgcount):
    for j in range(row):
        for k in range(col):
            img = cv2.imread(test_image[i])
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mattotal[j][k] += img1[j][k]

for j in range(row):
    for k in range(col):
        mattotal[j][k] *= 1/imgcount
        mattotal[j][k] = round(mattotal[j][k]/255) # bagi 255 biar nilainya kecil


selisih = [[0 for i in range(col)]for j in range(row)]
A = [[]for j in range(row)]
arr = []

for i in range(imgcount):
    for j in range(row):
        for k in range(col):
            img = cv2.imread(test_image[i])
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            selisih[j][k] = round(img1[j][k]/255 - mattotal[j][k]) # img1/255 biar nilainya kecil
    A = np.concatenate((A,selisih),axis=1)
    arr.append(selisih)

CovMat = cov.CovarianNumpy(A)
EigVec = np.linalg.eig(CovMat).round()

print(EigVec)