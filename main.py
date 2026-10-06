from PIL import Image
import numpy as np
import struct
import matplotlib.pyplot as plt
import math

# Re-sizing images to a 20*20 from an original 28*28 to reduce the cost matrix M size ==> faster iterations


# Input : Image file path  (train-images.idx3-ubyte)

# output : Normlaized histograms (matrix N*400, N images each 20*20=400), Cost Matrix (400*400, image vs image)
N_train = [3000, 5000, 8000, 12000, 17000, 25000]

for i in N_train:
    X_train = normalized_histogram[:i] # shape (N_train[i], 400)
    X_test = normalized_histogram[i:] # shape (len(N_train)-X_train.shape(0), 400)

median_M = np.median(cost_matrix_M)
lmbdas = 9/median_M # to be changed later

# transposing the X_train s.t. each image is a column
C = X_train.T

# computing the distaace r to the images in C (c1, c2, c3, ..., c3000)
test_dis = []

train_dist = []
test_dist = [] 


# train to train distances
for i in X_train: 
    sinkhorn_dist = ComputeSinkhornDistances(r, C, M, lmbdas, max_iter)
    train_dist.append(sinkhorn_dist)


# test to train distances
for i in X_test: 
    sinkhorn_dist = ComputeSinkhornDistances(r, C, M, lmbdas, max_iter)
    test_dist.append(sinkhorn_dist)
    


# possible values of t to be examined for the similarity score (large t : lower exponent means higher similarity score vice versa)

dist_flat = train_dist.flatten()
q1 = np.quantile(dist_flat, 0.1)
q2 = np.quantile(dist_flat, 0.2)
q5 = np.quantile(dist_flat, 0.5)

t_values = [1, q1,q2,q5]

best_t = CV_ct(t_values, c_values) # fix s.t. u only get best_t from the function
best_c = CV_ct(t_values, c_values) # fix s.t. u only get best_c from the function

def CV_ct(t_values, c_values):
    return best_t, best_c


# get the similarity scores by calculating the exponential function for both the training and test phase
train_kernel = math.exp(-train_dist/best_t)
test_kernel = math.exp(-test_dist/best_t)


# adding a value of 10^-5 to the diagonal values of the train_kernel to make sure it is psd and the SVM converges
train_kernel = train_kernel + 0.00001*(np.identity(N_train))


def ComputeSinkhornDistances(r, C, M, lmbdas, max_iter):
    K = np.exp(-lmbdas * M)
    N = C.shape[1] # number of images

    # u initialization
    u = np.ones((400, N))/400.0 # calculating the distance between r to c1, c2, ..., cN and keeping track of the u_n's

    for _ in range(max_iter):
        # calculating v_j = C_j/(K_T u)_j
        K_trans = np.dot(K.T, u)
        #element wise division
        V = C / K_trans  # V is 400*N

        # calculating u = r / (Kv)
        K_v = np.dot(K, V)  # K_v = 400*N since K = 400*400
        #element wise division
        u = r.reshape(-1, 1) / K_v
    #  our initial optimization problem is : P_ij*M_ij = u_i*K_ij*v_j*M_ij = u_i*K_ij*M_ij*v_j
    # now we compute K*M since we have the u and v 
    KM = K * M

    KMV = np.dot(KM, V)
    UKMV = u * KMV

    dist = np.sum(UKMV, axis = 0)

    return dist


image_file_path = "MNIST/train-images.idx3-ubyte" 
with open(image_file_path, "rb") as f: 
    # read bytes 0-3
    magic_number_bytes = struct.unpack(">I",f.read(4))[0]
    # read bytes 4-7 (automatically recalls to start from where it left off, i.e. byte 4)
    read_number_of_images = struct.unpack(">I", f.read(4))[0]
    read_rows = struct.unpack(">I", f.read(4))[0]
    read_cols = struct.unpack(">I", f.read(4))[0]

    # byte 16+ represents the pixel data (N * 28 * 28 bytes)
    # 8 bits = 1 byte, construct an array consisting of 1 byte (8 bits) in each element 
    raw_pixel_data = np.frombuffer(f.read(), dtype = np.uint8)
    images = raw_pixel_data.reshape((read_number_of_images, 28, 28))


# // Now we have the images, we are going to resize to 20*20 and project to the probability simplex (normalization)

normalized_histogram = []

for image in range(read_number_of_images):
    # we go from 28*28 array to image then resize to a 20*20 image, and then convert it back to an array 1D
    array_to_image = Image.fromarray(images[image])
    res = array_to_image.resize((20,20))
    flatt_array = np.array(res, dtype=np.float64).flatten() #flatten returns a 1D array of size 1*400 

    pixel_sum = flatt_array.sum()
    norm_image = flatt_array / pixel_sum # Image normalized to 1, probability distribution and added to the list

    normalized_histogram.append(norm_image)

print(normalized_histogram[0])


# initialize cost matrix M
cost_matrix_M = np.zeros((400, 400))

# initialize (x, y) coordinates
coordinates = []
for x in range(20): 
    for y in range(20):
        coordinates.append((x, y)) # [(0, 0), (0, 1), ...]


for i in range(400):
    for j in range(400): 
        # we calculate the euclidean distance between pixel i and j and then add it the cost matrix at M_ij
        x_d = coordinates[i][0] - coordinates[j][0]
        y_d = coordinates[i][1] - coordinates[j][1]

        cost_matrix_M[i, j] = math.sqrt(x_d**2 + y_d**2) 




<<<<<<< Updated upstream

=======
# computing the distaace r to the images in C (c1, c2, c3, ..., c3000)
test_dis = []
>>>>>>> Stashed changes
