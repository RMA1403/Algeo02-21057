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


def eigenvalues(M):
    """Fungsi untuk menghitung nilai-nilai eigen dari sebuah matriks"""
    # Validasi input terhadap matriks yang tidak persegi
    m, n = M.shape
    if m != n:
        return "Nilai eigen tidak terdefinisi untuk matriks non-persegi"

    # Dekomposisi QR awal
    Q, R = qr_decomposition(M)

    # Melakukan perkalian matriks R dan Q hingga didapat matriks diagonal
    hasEigen = False
    for _ in range(500):
        A = R @ Q
        if np.allclose(A, np.triu(A), 0.0001):
            hasEigen = True
            break
        Q, R = qr_decomposition(A)

    # Mengembalikan elemen-elemen diagonal utama yang merupakan nilai-nilai eigen dari matriks
    if (hasEigen):
        eigens = []
        for i in range(M.shape[0]):
            if A[i][i] not in eigens and np.round(A[i][i]) != 0.0:
                eigens.append(int(np.round(A[i][i])))
        return eigens
    else:
        return "Matriks tidak memiliki nilai eigen"


# Test case untuk beberapa matriks

# Matriks 2x2
A = np.array([[3, 0], [8, -1]])
print("Matrix A:\n", A)
print("Nilai eigen dari A:", eigenvalues(A))

# Matriks 2x2
B = np.array([[1, 3], [3, 1]])
print("\nMatrix B:\n", B)
print("Nilai eigen dari B:", eigenvalues(B))

# Matriks 3x3
C = np.array([[3, -2, 0], [-2, 3, 0], [0, 0, 5]])
print("\nMatrix C:\n", C)
print("Nilai eigen dari C:", eigenvalues(C))

# Matriks dengan nilai eigen 0
D = np.array([[10, 0, 2], [0, 10, 4], [2, 4, 2]])
print("\nMatrix D:\n", D)
print("Nilai eigen dari D:", eigenvalues(D))

# Matriks yang tidak memiliki nilai eigen
E = np.array([[-2, -1], [5, 2]])
print("\nMatrix E:\n", E)
print("Nilai eigen dari E:", eigenvalues(E))

# Matriks yang bukan persegi
F = np.array([[-2, -1, 3], [5, 2, 8]])
print("\nMatrix F:\n", F)
print("Nilai eigen dari F:", eigenvalues(F))
