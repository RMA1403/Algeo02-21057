import cv2
import numpy as np
from glob import glob
import Eigenvalue as eigval
import mean_selisih as mean

# Import dataset
dataset_folder = "C:/Users/Rava Maulana/Desktop/Test Data"
dataset_folder = glob(dataset_folder + "/*.png") + glob(dataset_folder + "/*.jpg")

# Change dataset image into N^2 x 1 vector
dataset = []
for i in range(len(dataset_folder)):
  dataset_image = cv2.imread(dataset_folder[i])
  dataset_image = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2GRAY)
  dataset_image = cv2.resize(dataset_image, (256, 256))
  dataset_image = dataset_image.reshape(256**2, 1)
  dataset.append(dataset_image)

# Computes mean face of the dataset
mean_face = mean.mean_selisih(dataset)[1]
mean_face = mean_face.astype("float64")

# Computes difference between each image and mean face
# temp_diff = dataset[i] - mean_face
# difference = temp_diff 
# for i in range(1, len(dataset)):
#   temp_diff = dataset[i] - mean_face
#   difference = np.concatenate((difference, temp_diff), axis = 1)
difference = mean.mean_selisih(dataset)[0]
difference = difference.astype("float64")

# Computes the simplified covariance matrix
simp_covariance = np.transpose(difference) @ difference
simp_covariance /= len(dataset)
simp_covariance /= len(dataset)
simp_covariance /= len(dataset)
simp_covariance /= len(dataset)

# Computes the eigenvalue of simplified covariance matrix
# simp_eigens = np.linalg.eig(simp_covariance)[1]
simp_eigens = eigval.eigen(simp_covariance)
print(simp_eigens.shape)

# Computes the eigenvalues from simplified eigenvalues
curr_simp_eigen = simp_eigens[0:, 0]
eigens = difference @ curr_simp_eigen.reshape(len(curr_simp_eigen), 1)
for i in range(1, simp_eigens.shape[1]):
  curr_simp_eigen = simp_eigens[0:, i]
  eigens = np.concatenate((eigens, difference @ curr_simp_eigen.reshape(len(curr_simp_eigen), 1)), axis=1)

# Normalize the eigenvalues to compute the eigenfaces
curr_eigen = eigens[0:, 0].reshape(len(eigens[0:, 0]), 1)
eigenfaces = curr_eigen/np.linalg.norm(curr_eigen)
for i in range(1, eigens.shape[1]):
  curr_eigen = eigens[0:, i].reshape(len(eigens[0:, i]), 1)
  eigenfaces = np.concatenate((eigenfaces, curr_eigen/np.linalg.norm(curr_eigen)), axis=1)

# Computes weight coefficients of each image in dataset
weights = np.transpose(eigenfaces) @ difference

# Reconstruction
curr_weight = weights[0:, 5]
curr_eigenface = eigenfaces[0:, 0].reshape(len(eigenfaces[0:, 0]), 1)
reconstruction = curr_weight[0] * curr_eigenface
for i in range(1, eigenfaces.shape[1]):
  curr_eigenface = eigenfaces[0:, i].reshape(len(eigenfaces[0:, i]), 1)
  reconstruction += curr_weight[i] * curr_eigenface
reconstruction += mean_face

recon_min = 999999
recon_max = -999999
for i in range(reconstruction.shape[0]):
  if reconstruction[i][0] < recon_min:
    recon_min = reconstruction[i][0]
  if reconstruction[i][0] > recon_max:
    recon_max = reconstruction[i][0]
recon_range = recon_max - recon_min
reconstruction = np.array([[(reconstruction[i][0] - recon_min)/recon_range * 255] for i in range(reconstruction.shape[0])])
reconstruction = reconstruction.reshape(256, 256)
reconstruction = np.around(reconstruction)
reconstruction = reconstruction.astype("uint8")

# print(curr_weight)
cv2.imshow("Testing", reconstruction)
cv2.waitKey(0)

