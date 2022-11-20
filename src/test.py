import cv2
import numpy as np
from glob import glob
import time

import Covarian as cov
import Eigenvalue as eigval
import EucDistance as eucdis
import mean_selisih as Mean

path = "./testImage"
test_image = glob(path + "/*.png") + glob(path + "/*.jpg")#
imgcount = len(test_image)
imglist = []
for i in range(len(test_image)):
    img = cv2.resize(cv2.cvtColor(cv2.imread(test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
    img = img.reshape(len(img)**2,1)
    imglist.append(img)

#print(np.asarray(imglist))
imgarr = np.asarray(imglist)


mean = Mean.mean_selisih(imgarr)[1]
selisih = Mean.mean_selisih(imgarr)[2]


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

print(eigfacearr[0])