import cv2
import numpy as np
from glob import glob
import functions.Eigen as eigval
import functions.MeanSelisih as mean
import os.path
import time
import math

async def train_dataset(dataset_path):
  start_time = time.time()
  # Import dataset
  dataset_folder = dataset_path

  # Checks for a cache file
  if os.path.exists(dataset_folder + "/dataset_weights_cache.txt") and os.path.exists(dataset_folder + "/eigenfaces_cache.txt") and os.path.exists(dataset_folder + "/meanface_cache.txt"):
    weight_arr = []
    with open(dataset_folder + "/dataset_weights_cache.txt", 'r') as f:
      for line in f:
        weight_arr.append(line.split())
    weights = np.array(weight_arr, dtype="float64")
    weights = np.transpose(weights)

    eigenfaces_arr = []
    with open(dataset_folder + "/eigenfaces_cache.txt", 'r') as f:
      for line in f:
        eigenfaces_arr.append(line.split())
    eigenfaces = np.array(eigenfaces_arr, dtype="float64")
    eigenfaces = np.transpose(eigenfaces)

    meanface_arr = []
    with open(dataset_folder + "/meanface_cache.txt", 'r') as f:
      for line in f:
        meanface_arr.append(line.split())
    mean_face = np.array(meanface_arr, dtype="float64")
    mean_face = np.transpose(mean_face)

    return weights, eigenfaces, mean_face, (time.time() - start_time)

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
  with open(dataset_path + "/dataset_weights_cache.txt", 'w') as f:
    for i in range(weights.shape[1]):
      weight_str = ""
      for j in range(len(weights[0:, i])):
        weight_str += str(weights[0:, i][j]) + " "
      weight_str += "\n"
      f.write(weight_str)

  # Store the eigenfaces in a cache file
  with open(dataset_path + "/eigenfaces_cache.txt", 'w') as f:
    for i in range(eigenfaces.shape[1]):
      eigenfaces_str = ""
      for j in range(len(eigenfaces[0:, i])):
        eigenfaces_str += str(eigenfaces[0:, i][j]) + " "
      eigenfaces_str += "\n"
      f.write(eigenfaces_str)

  # Store the mean face in a cache file
  with open(dataset_path + "/meanface_cache.txt", 'w') as f:
    meanface_str = ""
    for i in range(len(mean_face[0:, 0])):
      meanface_str += str(mean_face[0:, 0][i]) + " "
    meanface_str += "\n"
    f.write(meanface_str)
  
  return weights, eigenfaces, mean_face, (time.time() - start_time)

async def match_image(image_path, dataset_weights, eigenfaces, mean_face, dataset):
  start_time = time.time()
  # Import file gambar
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (256, 256))
  image = image.reshape(256**2, 1)

  # Menghitung selisih dari gambar
  difference = image - mean_face

  # Menghitung bobot dari setiap eigenface
  weights = np.transpose(eigenfaces) @ difference

  # Mencari euclidian distance terkecil
  idx_min = 0
  euc_distance = 0.0
  for i in range(dataset_weights.shape[0]):
    euc_distance += (weights[0:, 0][i] - dataset_weights[0:, 0][i])**2
  euc_distance = math.sqrt(euc_distance)
  euc_min = euc_distance

  for i in range(1, dataset_weights.shape[1]):
    euc_distance = 0.0
    for j in range(dataset_weights.shape[0]):
      curr_dataset_weights = dataset_weights[0:, i]
      euc_distance += (weights[0:, 0][j] - curr_dataset_weights[j])**2
    euc_distance = math.sqrt(euc_distance)
    if euc_distance < euc_min:
      euc_min = euc_distance
      idx = i

  output_image = (glob(dataset + "/*.png") + glob(dataset + "/*.jpg"))[idx]
  output_image = output_image.replace("\\", "/")
  return euc_min, output_image, (time.time() - start_time)
