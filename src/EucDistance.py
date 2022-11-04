import numpy as np

# Membuat matriks eigenface test face dengan eigenvector dan matriks selisih matriks testface dengan matriks mean training image
# X : eigenvector
# Y : matriks test face
# Z : mean training image
def newFaceMatNumpy(X, Y, Z):
    
    selisih = np.subtract(Y,Z)
    eigennew = np.matmul(X,selisih)

    # print(eigennew)

    return eigennew

def newFaceMatBase(X, Y, Z):

    row1 = len(Y)
    col1 = len(Y[0])

    selisih = [[0 for i in range(col1)]for j in range(row1)]

    for i in range(row1):
        for j in range(col1):
            selisih[i][j] = Y[i][j] - Z[i][j]

    row2 = len(selisih)
    col2 = len(selisih[0])

    eigennew = [[0 for i in range(col1)]for j in range(row1)]

    for k in range(row1):
        for l in range(col2):
            sum = 0
            for m in range(row2):
                sum += X[k][m] * selisih[m][l]
            eigennew[k][l] = sum

    # print(eigennew)

    return eigennew

# Selisih eigenface foto test face dengan training image
# new : matriks eigenface test face
# test : matriks eigenface training image
def EucDistanceNumpy(new,test):

    newmat = np.subtract(new,test)

    row = len(newmat)
    col = len(newmat[0])

    count = 0
    for i in range(row):
        for j in range(col):
            count += (newmat[i][j])**2

    # print(count)

    distance = np.sqrt(count)

    # print(distance)

    return distance

def EucDistanceBase(new,test):

    row = len(new)
    col = len(new[0])

    newmat = [[0 for i in range(col)]for j in range(row)]

    for i in range(row):
        for j in range(col):
            newmat[i][j] = new[i][j] - test[i][j]

    count = 0
    for i in range(row):
        for j in range(col):
            count += (newmat[i][j])**2

    # print(count)

    distance = count**(0.5)

    # print(distance)

    return distance

X = [[1,-1,0],
    [-1,-1,0],
    [0,0,0]]

Y = [[2,1,1],
    [1,2,1],
    [0,2,4]]

Z = [[1,0,1],
    [0,1,0],
    [0,2,3]]

test1 = [[0,-1,0],
        [-2,-1,0],
        [0,0,0]]

test2 = [[0,1,0],
        [0,-1,0],
        [0,0,0]]

eigennew = newFaceMatNumpy(X,Y,Z)
EucDistanceBase(eigennew,test2)