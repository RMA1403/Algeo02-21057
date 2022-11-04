import numpy as np

def CovarianNumpy(X):

    Y = np.transpose(X)

    Covarian = np.matmul(X,Y)

    # print(Covarian)
    
    return Covarian

def CovarianBase(X):

    row1 = len(X)
    col1 = len(X[0])

    Y = [[0 for i in range(row1)] for j in range(col1)]

    for r in range(col1):
        for c in range(row1):
            Y[r][c] = X[c][r]
    
    # print(Y)

    row2 = len(Y)
    col2 = len(Y[0])

    Covarian = [[0 for i in range(col2)] for j in range(row1)]
    
    for k in range(row1):
        for l in range(col2):
            sum = 0
            for m in range(row2):
                sum += X[k][m] * Y[m][l]
            Covarian[k][l] = sum

    # print(Covarian)
    
    return Covarian

# Data buat Test
X = [[1,0,0,0,1,0],
    [1,1,0,0,0,0],
    [0,0,1,1,0,1]]

X1 = [[1, 1, 0], 
    [0, 1, 0], 
    [0, 0, 1], 
    [0, 0, 1], 
    [1, 0, 0], 
    [0, 0, 1]]

Y = [[1,2,3],
    [4,5,6],
    [7,8,9]]

A = [[1, 4, 7], 
    [2, 5, 8], 
    [3, 6, 9]]
    
CovarianNumpy(X)