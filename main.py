# from PIL import images
# import numpy as np
import struct

# Re-sizing images to a 20*20 from an original 28*28 to reduce the cost matrix M size ==> faster iterations


# Input : Image file path  (train-images.idx3-ubyte)

# output : Normlaized histograms (matrix N*400, N images each 20*20=400), Cost Matrix (400*400, image vs image)



image_file_path = "MNIST/train-images.idx3-ubyte" 
with open(image_file_path, "rb") as f: 
    # read bytes 0-3
    magic_number_bytes = struct.unpack(">I",f.read(4))
    # read bytes 4-7 (automatically recalls to start from where it left off, i.e. byte 4)
    read_number_of_images = struct.unpack(">I", f.read(4))
    read_rows = struct.unpack(">I", f.read(4))
    read_cols = struct.unpack(">I", f.read(4))
    print(magic_number_bytes)
    
