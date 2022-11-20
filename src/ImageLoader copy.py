import cv2
import numpy as np
from glob import glob
import time

import Covarian as cov
import Eigenvalue as eigval
import EucDistance as eucdis
import mean_selisih as Mean

async def loadImage(testface, test_image):
    start_time = time.time()
    print("enter")
    # Membaca training face
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

    # Menentukan eigen value dari matriks Covarian dan membentuk eigen vectornya
    EigVec = np.array(eigval.eigen(CovMat))

    # Menentukan eigenface dari semua training image
    eigenface = []
    for i in range(len(EigVec)):
        eigenface.append(np.asarray((np.matmul(A,EigVec[i]))))

    eigfacearr = np.asarray(eigenface)

    # Membaca test face dengan ukuran diubah menjadi 256 x 256 
    # dan nilai tiap elemennya dibagi 255, kemudian ditentukan eigenfacenya
    sampleimage = cv2.resize(cv2.cvtColor(
        (cv2.imread(testface)), cv2.COLOR_BGR2GRAY), (256, 256))
    sampleimage = sampleimage / 255
    selisihsam = abs(sampleimage - mean)
    EigFaceSam = np.matmul(EigVec, selisihsam)

    # Menghitung euclidean distance dari test face dengan semua training image
    eucdisarr = []
    for i in range(len(eigfacearr)):
        eucdisarr.append(eucdis.EucDistanceNumpy(EigFaceSam, eigfacearr[i]))

    # Menentukan euclidean distance terkecil dari test face
    min = eucdisarr[0]
    idx = 0

    for i in range(len(eucdisarr)):
        if min > eucdisarr[i]:
            idx = i
            min = eucdisarr[i]

    # Mengembalikan nama file dari training image yang mendekati test face dan
    # euclidean distancenya, serta lama waktu pemrosesan
    return test_image[idx], eucdisarr[idx], (time.time() - start_time)