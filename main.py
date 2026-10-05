from PIL import Image
import numpy as np
import struct
import matplotlib.pyplot as plt
import math

# Re-sizing images to a 20*20 from an original 28*28 to reduce the cost matrix M size ==> faster iterations


# Input : Image file path  (train-images.idx3-ubyte)

# output : Normlaized histograms (matrix N*400, N images each 20*20=400), Cost Matrix (400*400, image vs image)


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

for image in range(read_number_of_images-1):
    # we go from 28*28 array to image then resize to a 20*20 image, and then convert it back to an array 1D
    array_to_image = Image.fromarray(images[image])
    res = array_to_image.resize((20,20))
    flatt_array = np.array(res, dtype=np.float64).flatten() #flatten returns a 1D array of size 1*400 

    pixel_sum = flatt_array.sum()
    norm_image = flatt_array / pixel_sum # Image normalized to 1, probability distribution and added to the list

    normalized_histogram.append(norm_image)

print(flatt_array)


# initialize cost matrix M
cost_matrix_M = np.zeros(400, 400)

# initialize (x, y) coordinates
coordinates = []

#for i in range(400):
#    for j in range(400): 
        # we calculate the euclidean distance between pixel i and j and then add it the cost matrix at M_ij
#        x_d = 
