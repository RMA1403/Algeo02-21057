import numpy as np


def householder_transformation(x):
    """Transformasi householder"""
    v = x / (x[0]+np.copysign(np.linalg.norm(x), x[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)
    return v, tau


def qr_decomposition(M):
    """Fungsi dekomposisi QR menggunakan transformasi householder"""
    m, n = M.shape
    R = M.copy()
    Q = np.identity(m)

    for i in range(0, n):
        v, tau = householder_transformation(R[i:, i, np.newaxis])
        H = np.identity(m)
        H[i:, i:] -= tau * (v @ v.T)
        R = H @ R
        Q = H @ Q

    return Q[:n].T, np.triu(R[:n])


def eigen(M):
    """Fungsi untuk menghitung nilai eigen dan vektor eigen dari sebuah matriks"""
    # Validasi input terhadap matriks yang tidak persegi
    m, n = M.shape
    if m != n:
        return "Nilai eigen tidak terdefinisi untuk matriks non-persegi"

    # Dekomposisi QR awal
    Q, R = qr_decomposition(M)
    eigens = Q

    # Melakukan perkalian matriks R dan Q hingga didapat matriks diagonal
    hasEigen = False
    for i in range(500):
        A = R @ Q

        isEqual = True
        for i in range(A.shape[0]):
            if (not isEqual):
                break
            for j in range(A.shape[1]):
                if i != j and A[i][j] > 0.1:
                    isEqual = False
                    break

        if isEqual:
            hasEigen = True
            break
        Q, R = qr_decomposition(A)
        eigens = eigens @ Q

    # Mengembalikan nilai eigen dan vektor eigen dari matriks
    if (hasEigen):
        for i in range(len(eigens)):
            for j in range(len(eigens[0])):
                eigens[i][j] = np.round(eigens[i][j], 3)
        # print(eigens)
        eigens = np.hsplit(eigens, eigens.shape[1])
        return eigens
    else:
        return "Matriks tidak memiliki nilai eigen"


# Test case untuk beberapa matriks

# Matriks 2x2
# A = np.array([[3, 0], [8, -1]])
# print("Matrix A:\n", A)
# print("Nilai eigen dari A:\n", eigen(A)[0], "\n" ,eigen(A)[1])
# print(np.linalg.eig(A)[1])

# # Matriks 2x2
# B = np.array([[1, 3], [3, 1]])
# print("\nMatrix B:\n", B)
# print("Nilai eigen dari B:", eigen(B)[1])
# print(np.linalg.eig(B)[1])

# # Matriks 3x3
# C = np.array([[3, -2, 0], [-2, 3, 0], [0, 0, 5]])
# print("\nMatrix C:\n", C)
# print("Nilai eigen dari C:", eigen(C)[1])
# print(np.linalg.eig(C)[1])

# # Matriks dengan nilai eigen 0
# D = np.array([[10, 0, 2], [0, 10, 4], [2, 4, 2]])
# print("\nMatrix D:\n", D)
# print("Nilai eigen dari D:", eigen(D)[1])
# print(np.linalg.eig(D)[1])

# # Matriks yang tidak memiliki nilai eigen
# E = np.array([[-2, -1], [5, 2]])
# print("\nMatrix E:\n", E)
# print("Nilai eigen dari E:", eigen(E))

# # Matriks yang bukan persegi
# F = np.array([[-2, -1, 3], [5, 2, 8]])
# print("\nMatrix F:\n", F)
# print("Nilai eigen dari F:", eigen(F))
