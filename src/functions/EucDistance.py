import numpy as np

# Selisih eigenface foto test face dengan training image
# new : matriks eigenface test face
# test : matriks eigenface training image
def EucDistanceNumpy(new,test):

    newmat = np.subtract(np.array(new),np.array(test))

    row = len(newmat)

    count = 0
    for i in range(row):
        count += (newmat[i][0])**2

    distance = np.sqrt(count)

    return distance

def EucDistanceBase(new,test):

    row = len(new)
    col = len(new[0])

    newmat = [[0 for i in range(col)]for j in range(row)]

    for i in range(row):
        newmat[i][0] = new[i][0] - test[i][0]

    count = 0
    for i in range(row):
        count += (newmat[i][0])**2

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

# print(EucDistanceBase(test1,test2))