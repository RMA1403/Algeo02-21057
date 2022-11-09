import numpy as np

#dataset
m1 = np.array([[2,0,1], [1,2,0], [0,2,4]])
m2 = np.array([[1,1,1], [0,1,0], [1,2,2]])
mock_matrix = np.array([[1,-1,0], [-1,-1,0], [0,0,0]])
row1 = len(m1)
col1 = len(m1[0])
m = [[0 for i in range(col1)]for j in range(row1)]
arr = [m1, m2]
i = 0


def matrixAdd (m1):
    row1 = len(m1)
    col1 = len(m1[0])
    m = [[0 for i in range(col1)]for j in range(row1)]
    
    m += m1
    return m
    

def mean (array):
    row = len(array[0]) #baca baris dari matrix pertama
    col = len(array[0][0]) #baca kolom dari matrix pertama
    m = [[0 for i in range(col)]for j in range(row)] #membuat matrix berelemen nol dengan ukuran row1 x col1 
    count = 0

    if m1.shape != m2.shape:
        return False
    else:
        while count < len(array):
            m += matrixAdd(array[count]) #menambahkan semua matrix
            count += 1
        
        for i in range (row):
            for j in range (col):
                m[i][j] = np.floor(1/count * m[i][j]) #mengalikan hasil akhir matrix dengan 1/M
        
        return m

def selisih(array, mean):
    selisih = []
    for i in range (len(array)):
        selisih.append(np.subtract(array[i], mean))


    
             
    
    return array1
    

def eigenface(array, eigenvec):
    array2 = [0 for i in range(len(array))]
    arr = selisih(array)

    for i in range(len(array)):
        array2[i] = np.matmul(eigenvec,arr[i])
    
    return array2

# print(eigenface(arr, mock_matrix))


#newmat = np.array([[1,2,3], [4,5,6]])

#row = len(newmat)
#col = len(newmat[0])

#print(newmat)
#print(row)
#print(col)