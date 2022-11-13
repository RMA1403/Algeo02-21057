import numpy as np

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

    distance = np.sqrt(count)

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

    distance = count**(0.5)

    return distance

# X = [[1,-1,0],
#     [-1,-1,0],
#     [0,0,0]]

# Y = [[2,1,1],
#     [1,2,1],
#     [0,2,4]]

# Z = [[1,0,1],
#     [0,1,0],
#     [0,2,3]]

# test1 = [[0,-1,0],
#         [-2,-1,0],
#         [0,0,0]]

# test2 = [[0,1,0],
#         [0,-1,0],
#         [0,0,0]]

# eigennew = newFaceMatNumpy(X,Y,Z)
# EucDistanceBase(eigennew,test2)