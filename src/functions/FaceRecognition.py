import cv2
import numpy as np
from glob import glob
import functions.Eigen as eigval
import functions.MeanSelisih as mean
from os.path import exists

async def train_dataset(dataset_path):
  # Import dataset
  dataset_folder = dataset_path
  
  # Checks for a cache file
  if exists(dataset_folder + "/eigen_cache.txt"):
    weight_arr = []
    with open(dataset_folder + "/eigen_cache.txt", 'r') as f:
      for line in f:
        weight_arr.append(line.split())
    weights = np.array(weight_arr, dtype="float64")
    return weights

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
  difference = mean.mean_selisih(dataset)[0]
  difference = difference.astype("float64")

  # Computes the simplified covariance matrix
  simp_covariance = np.transpose(difference) @ difference
  simp_covariance /= len(dataset)
  simp_covariance /= len(dataset)
  simp_covariance /= len(dataset)
  simp_covariance /= len(dataset)

  # Computes the eigenvalue of simplified covariance matrix
  simp_eigens = eigval.eigen(simp_covariance)

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

  # Store the weights in a cache file
  with open(dataset_path + "/eigen_cache.txt", 'w') as f:
    for i in range(weights.shape[1]):
      weight_str = ""
      for j in range(len(weights[0:, i])):
        weight_str += str(weights[0:, i][j]) + " "
      weight_str += "\n"
      f.write(weight_str)
  
  return weights
