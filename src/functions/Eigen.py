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
        return eigens
    else:
        return "Matriks tidak memiliki nilai eigen"
