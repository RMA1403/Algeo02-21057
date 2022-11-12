import cv2
import numpy as np
from glob import glob
import matplotlib as plt

import Covarian as cov
import Eigenvalue as eigval
import Eigenvector as eigvec
import EucDistance as eucdis
import mean

test_image = glob("./test/testImage/*.png") + glob("./test/testImage/*.jpg")

imgcount = len(test_image)

imglist = []
for i in range(imgcount):
    img = cv2.resize(cv2.cvtColor(cv2.imread(
        test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
    img = img / 255
    imglist.append(img)

imgarr = np.asarray(imglist)
mean = mean.mean(imgarr)
selisih = [0 for i in range(len(imgarr))]
for i in range(len(imgarr)):
    selisih[i] = abs(imgarr[i] - mean)

A = [[]for j in range(len(selisih[0]))]

for i in range(len(selisih)):
    A = np.concatenate((A, selisih[i]), axis=1)

CovMat = cov.CovarianNumpy(A)
EigVec = np.array(eigval.eigen(CovMat)[1])

eigenface = []
for i in range(len(selisih)):
    eigenface.append(np.asarray((np.matmul(EigVec, selisih[i]))))

eigfacearr = np.asarray(eigenface)

testface = glob("./test/testImage/testFace/*.png") + \
    glob("./test/testImage/testFace/*.jpg")
sampleimage = cv2.resize(cv2.cvtColor(
    (cv2.imread(testface[0])), cv2.COLOR_BGR2GRAY), (256, 256))
sampleimage = sampleimage / 255
selisihsam = abs(sampleimage - mean)
EigFaceSam = np.matmul(EigVec, selisihsam)

row = len(EigFaceSam)
col = len(EigFaceSam[0])


eucdisarr = []
for i in range(len(eigfacearr)):
    eucdisarr.append(eucdis.EucDistanceNumpy(EigFaceSam, eigfacearr[i]))

min = eucdisarr[0]
idx = 0

for i in range(len(eucdisarr)):
    if min > eucdisarr[i]:
        idx = i
        min = eucdisarr[i]

print(test_image[idx], eucdisarr[idx])
