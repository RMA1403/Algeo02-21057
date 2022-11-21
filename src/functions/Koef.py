import numpy as np

# Menghasilkan matriks koefisien eigenface dari matriks eigenface dan matriks selisih
# arrEigFace = array yang memiliki elemen matriks eigenface
# arrSelisih = array yang memiliki elemen matriks selisih
def allMatKoef(arrEigFace, arrSelisih):

    allImage = len(arrSelisih)
    allEigFace = len(arrEigFace)

    arrMatKoef = []
    for i in range(allImage):
        curr_image = arrSelisih[i]
        MatKoef = []
        for j in range(allEigFace):
            curr_eigFace = np.array(np.transpose(arrEigFace[j]))
            Koef = np.matmul(curr_eigFace,curr_image)
            MatKoef.append(Koef[0][0])
        currMatKoef = np.asarray(np.array(MatKoef).reshape(len(MatKoef),1))
        np.asarray(arrMatKoef.append(currMatKoef))

    return np.asarray(arrMatKoef)


def findNorm(eigVec):
    
    row = len(eigVec)

    total = 0
    for i in range(row):
        total += eigVec[i][0]**2
    
    Norm = total**0.5
    return Norm

# X = [[[1],[2],[3]],[[4],[5],[6]]]
# Y = [[[0],[1],[2]],[[9],[10],[11]]]

# print((allMatKoef(X,Y)))