import numpy as np

# path = "./testImage"
# test_image = glob(path + "/*.png") + glob(path + "/*.jpg")#
# imgcount = len(test_image)
# imglist = []
# for i in range(len(test_image)):
#     img = cv2.resize(cv2.cvtColor(cv2.imread(test_image[i]), cv2.COLOR_BGR2GRAY), (256, 256))
#     img = img.reshape(len(img)**2,1)
#     imglist.append(img)

# print(np.asarray(imglist))
# imgarr = np.asarray(imglist)

def mean_selisih(array):
    row = len(array[0])  # baca baris dari matrix pertama
    col = len(array[0][0])
    m = [[0 for i in range(col)]for j in range(row)]
    count = 0

    while count < len(array):
        m += array[count] # menambahkan semua matrix
        count += 1

    for i in range(row):
        for j in range(col):
            # mengalikan hasil akhir matrix dengan 1/M
            m[i][j] = 1/count * m[i][j]

    matA = [[]for i in range(len(array[0]))]
    selisih = []
    
    for i in range(len(array)):
        matA = np.concatenate((matA,(array[i]-m)), axis = 1)
        selisih.append(array[i]-m)

    return matA, m, np.asarray(selisih)

# imgarr = [[[1],[2],[3]],[[1],[2],[3]]]

# print(mean_selisih(imgarr))