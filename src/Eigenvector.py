import numpy as np
import Eigenvalue


def null_space(A):
    """Fungsi untuk menentukan basis ruang solusi Ax = 0"""
    # Menyalin matriks dan mengubah tipe datanya menjadi float
    row, col = A.shape
    M = []
    for i in range(row):
        temp = []
        for j in range(col):
            temp.append(A[i, j]*1.0)
        M.append(temp)
    col += 1
    for i in range(row):
        M[i].append(0.0)

    # Membentuk matriks eselon baris
    currentRow = 0
    j = 0
    while j < col-1 and currentRow < row:
        i = currentRow + 1
        while i < row and M[currentRow][j] == 0.0:
            if M[i][j] != 0.0:
                # swap_row(currentRow, i, row, col, M)
                for k in range(col):
                    temp = M[currentRow][k]
                    M[currentRow][k] = M[i][k]
                    M[i][k] = temp
            i += 1

        if M[currentRow][j] != 0.0:
            tempVal = M[currentRow][j]
            for k in range(col):
                M[currentRow][k] /= tempVal
            M[currentRow][j] = 1.0

            for i in range((currentRow+1), row):
                rowArray = []
                for k in range(col):
                    rowArray.append(M[currentRow][k])

                for k in range(col):
                    rowArray[k] *= M[i][j]

                k = 0
                while k < len(rowArray) and k < col:
                    M[i][k] -= rowArray[k]
                    k += 1

                M[i][j] = 0.0

            currentRow += 1

        j += 1

    # Menentukan basis dari ruang null
    nonPivot = []
    for i in range(row):
        for j in range(col):
            if M[i][j] != 0.0:
                nonPivot.append(j)
                break

    nullBasis = []
    for i in range(col-1):
        temp = []
        for j in range(col-1):
            if j in nonPivot:
                temp.append(None)
            else:
                temp.append(0)
        nullBasis.append(temp)

    check = col-2
    for i in range(row-1, -1, -1):
        for j in range(col-1):
            if M[i][j] != 0.0:
                if j < check:
                    while check != j:
                        nullBasis[check][check] = 1.0
                        check -= 1
                for k in range(j+1, col-1):
                    if M[i][k] != 0.0 and nullBasis[j][k] != None:
                        nullBasis[j][k] = -1.0*M[i][k]
                check -= 1
                break

    return [[nullBasis[i][j] for j in range(col-1) if nullBasis[i][j] != None] for i in range(row)]


def eigenvectors(M):
    """Fungsi untuk menentukan vektor-vektor dari sebuah matriks"""
    eig_values = Eigenvalue.eigenvalues(M)
    # Validasi input terhadap matriks
    if eig_values == "Matriks tidak memiliki nilai eigen":
        return eig_values
    elif eig_values == "Nilai eigen tidak terdefinisi untuk matriks non-persegi":
        return eig_values

    # Menghitung vektor-vektor eigen
    eigens = null_space(eig_values[0]*np.identity(M.shape[0]) - M)

    for i in range(1, len(eig_values)):
        eigens = np.concatenate((eigens, null_space(
            eig_values[i]*np.identity(M.shape[0]) - M)), axis=1)

    return eigens


# Test case untuk beberapa matriks

# Matriks 2x2
# A = np.array([[3, 0], [8, -1]])
# print("Matrix A:\n", A)
# print("Vektor eigen dari A:")
# print(eigenvectors(A))

# # Matriks 2x2
# B = np.array([[1, 3], [3, 1]])
# print("\nMatrix B:\n", B)
# print("Vektor eigen dari B:")
# print(eigenvectors(B))

# # Matriks 3x3
# C = np.array([[3, -2, 0], [-2, 3, 0], [0, 0, 5]])
# print("\nMatrix C:\n", C)
# print("Vektor eigen dari C:")
# print(eigenvectors(C))

# # Matriks dengan nilai eigen 0
# D = np.array([[10, 0, 2], [0, 10, 4], [2, 4, 2]])
# print("\nMatrix D:\n", D)
# print("Vektor eigen dari D:")
# print(eigenvectors(D))

# # Matriks yang tidak memiliki nilai eigen
# E = np.array([[-2, -1], [5, 2]])
# print("\nMatrix E:\n", E)
# print("Vektor eigen dari E:", eigenvectors(E))

# # Matriks yang bukan persegi
# F = np.array([[-2, -1, 3], [5, 2, 8]])
# print("\nMatrix F:\n", F)
# print("Vektor eigen dari F:", eigenvectors(F))

# X = np.array([[2,1,0],[1,2,0],[0,0,3]])
# print(null_space(X))