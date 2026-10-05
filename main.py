# from PIL import images
import numpy as np
import struct
import matplotlib.pyplot as plt

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


    # Assuming your array is named 'data'
    print(images.shape) # This will tell you the exact size: e.g., (10, 100, 50)

    # To visualize the first 2D "slice" of the 3D cube:
    plt.imshow(images[0, :, :], cmap='viridis')
    plt.colorbar()
    plt.title("Slice 0 of the 3D Array")
    plt.show()
# // Now we have the images, we are going to resize to 20*20 and project to the probability simplex (normalization)

normalized_histogram = []

#for image in images[images, :, :]:
#    image_obj = image
#    print(image)

print(images[0, :, :])

