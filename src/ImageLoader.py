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

imglist = []
for i in range(imgcount):
    img = cv2.imread(test_image[i])
    imglist.append()

selisih = mean.selisih(imglist)

# img1 = cv2.cvtColor(cv2.imread(test_image[0]), cv2.COLOR_BGR2GRAY)

# row = img1.shape[0]
# col = img1.shape[1]

# matmean = [[0 for i in range(col)]for j in range(row)]

# for i in range(imgcount):
#     for j in range(row):
#         for k in range(col):
#             img = cv2.imread(test_image[i])
#             img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             matmean[j][k] += img1[j][k]

# for j in range(row):
#     for k in range(col):
#         matmean[j][k] *= 1/imgcount
#         matmean[j][k] = round(matmean[j][k]) # bagi 255 biar nilainya kecil


# selisih = [[0 for i in range(col)]for j in range(row)]
# A = [[]for j in range(row)]
# arr = []

# for i in range(imgcount):
#     for j in range(row):
#         for k in range(col):
#             img = cv2.imread(test_image[i])
#             img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             selisih[j][k] = round(img1[j][k] - matmean[j][k]) # img1/255 biar nilainya kecil
#     A = np.concatenate((A,selisih),axis=1)
#     arr.append(selisih)

CovMat = cov.CovarianNumpy(A)
Eig = np.linalg.eig(CovMat)
EigVec = np.asmatrix(Eig[1])

EigFace = mean.eigenface(imglist,EigVec)

# EigFace = []
# for i in range(imgcount):
#     egface = np.matmul(EigVec,arr[i])
#     EigFace.append(egface)

testface = glob("./test/testImage/testFace/*.png")
sampleimage = cv2.cvtColor((cv2.imread(testface[0])), cv2.COLOR_BGR2GRAY)

selisihsample = np.asmatrix(np.subtract(sampleimage,mean.mean(imglist)))

eigtestface = np.matmul(EigVec, selisihsample)
currtestface = np.asmatrix(eigtestface)

print(currtestface)