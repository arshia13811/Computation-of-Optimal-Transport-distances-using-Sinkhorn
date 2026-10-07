import numpy as np
from config import *
from data_loader import find_idx, load_mnist, make_histograms
# import your SVM and Sinkhorn functions here later

def main():
    print("Locating MNIST files...")
    img_path = find_idx("train-images.idx3-ubyte", DATA_DIR)
    lbl_path = find_idx("train-labels.idx1-ubyte", DATA_DIR)
    
    print("Parsing binary data...")
    raw_images, labels = load_mnist(img_path, lbl_path)
    
    print(f"Loaded {len(raw_images)} images. Generating normalized histograms...")
    X_normalized = make_histograms(raw_images, side=SIDE)
    
    print(f"Setup complete. Matrix shape: {X_normalized.shape}")
    


if __name__ == "__main__":
    main()
