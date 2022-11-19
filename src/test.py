import cv2
import numpy as np
from glob import glob
import time

import Covarian as cov
import Eigenvalue as eigval
import EucDistance as eucdis
import mean_selisih as Mean
import Koef as kf
test_image = "./testimage"
test_image = glob(test_image + "/*.png") + glob(test_image + "/*.jpg")

# Memasukkan training face pada sebuah list dengan ukuran 
# diubah menjadi 256 x 256 dan tiap elemennya dibagi 255
imgcount = len(test_image)
imglist = []
for i in range(imgcount):
    img = cv2.resize(cv2.cvtColor(cv2.imread(
        test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
    img = img.reshape(len(img)**2,1) / 255
    imglist.append(img)

# Menentukan mean dan selisih dari training face
mean = Mean.mean_selisih(imglist)[1]
selisih = Mean.mean_selisih(imglist)[2]

# Menentukan matriks Covarian
A = np.asarray(Mean.mean_selisih(imglist)[0])
CovMat = cov.CovarianNumpy(A)
# print(CovMat)

# Menentukan eigen value dari matriks Covarian dan membentuk eigen vectornya
EigVec = np.array(eigval.eigen(CovMat))
# print(EigVec)

# Menentukan eigenface dari semua training image
eigenface = []
for i in range(len(EigVec)):
    eigenface.append(np.asarray(np.matmul(A,EigVec[i])))
eigfacearr = np.asarray(eigenface)

# image = np.array(mean).reshape((256, 256))
# cv2.imshow('Contoh',image)
# cv2.waitKey(0)

# Menentukan matriks koefisien
arrMatKoef = kf.allMatKoef(eigfacearr,selisih)
# print(EigVec)
# print(A[0],EigVec[0])
# print(eigfacearr[0],arrMatKoef[0])
arr = np.multiply(eigfacearr[0],arrMatKoef[0][0])
for i in range(1,len(eigfacearr)):
    arr += np.multiply(eigfacearr[i],arrMatKoef[0][i])
arr = np.array(arr).reshape(len(arr),1)
arr += mean
arr = np.array(arr).reshape((256,256))
print(arrMatKoef)
# cv2.imshow('Contoh',arr)
# cv2.waitKey(0)

# Membaca test face dengan ukuran diubah menjadi 256 x 256 
# dan nilai tiap elemennya dibagi 255, kemudian ditentukan eigenfacenya
testface = glob("./testimage/testFace/*.jpg")
sampleimage = cv2.resize(cv2.cvtColor(
    (cv2.imread(testface[0])), cv2.COLOR_BGR2GRAY), (256, 256))
sampleimage = sampleimage.reshape(len(sampleimage)**2,1) / 255
selisihsam = sampleimage - mean

eigfacelen = len(eigfacearr)
MatKoefSam = []
for j in range(eigfacelen):
        curr_eigFace = np.transpose(eigfacearr[i])
        SamKoef = np.matmul(curr_eigFace,selisihsam)
        MatKoefSam.append(SamKoef)
MatKoefSam = np.asarray(np.array(MatKoefSam).reshape(len(MatKoefSam),1))

# Menghitung euclidean distance dari test face dengan semua training image
eucdisarr = []
for i in range(len(arrMatKoef)):
    eucdisarr.append(eucdis.EucDistanceNumpy(MatKoefSam, arrMatKoef[i]))

# Menentukan euclidean distance terkecil dari test face
min = eucdisarr[0]
idx = 0

for i in range(len(eucdisarr)):
    if min > eucdisarr[i]:
        idx = i
        min = eucdisarr[i]