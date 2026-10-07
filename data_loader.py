import gzip
import struct
import numpy as np
from PIL import Image

def find_idx(stem, data_dir):
    # Check for standard and hyphenated names, with or without .gz compression
    for name in (stem, stem.replace(".", "-")):
        for suffix in ("", ".gz"):
            path = data_dir / (name + suffix)
            if path.is_file():
                return path
    raise FileNotFoundError(f"Make sure {stem} (or its .gz version) is in {data_dir}")

def load_mnist(image_path, label_path):
    opener = gzip.open if str(image_path).endswith(".gz") else open
    
    with opener(image_path, "rb") as f:
        img_bytes = f.read()
    with opener(label_path, "rb") as f:
        lbl_bytes = f.read()

    if len(img_bytes) < 16 or len(lbl_bytes) < 8:
        raise ValueError("IDX header is missing or corrupted.")

    # Unpack the binary headers
    magic, count, rows, cols = struct.unpack(">IIII", img_bytes[:16])
    label_magic, label_count = struct.unpack(">II", lbl_bytes[:8])

    if magic != 2051 or label_magic != 2049:
        raise ValueError("Magic numbers don't match the MNIST specification.")
    if count != label_count or count == 0:
        raise ValueError("Mismatched image and label counts.")

    # Extract the actual pixel and label data
    pixels = np.frombuffer(img_bytes, dtype=np.uint8, offset=16)
    labels = np.frombuffer(lbl_bytes, dtype=np.uint8, offset=8).astype(int)

    if np.any(labels > 9):
        raise ValueError("Labels must be integers between 0 and 9.")

    return pixels.reshape(count, rows, cols), labels

def make_histograms(images, side=20):
    histograms = []
    
    for img in images:
        # Resize image to reduce the cost matrix dimensions (faster iterations)
        resized = Image.fromarray(img).resize((side, side), resample=Image.Resampling.BICUBIC)
        flat = np.asarray(resized, dtype=np.float64).flatten()
        
        pixel_sum = flat.sum()
        if pixel_sum <= 0:
            raise ValueError("Found a completely black image; cannot normalize.")
        
        # Project onto the probability simplex (sums to 1)
        histograms.append(flat / pixel_sum)

    # Convert the final list into a 2D matrix so we can easily transpose it later
    return np.asarray(histograms, dtype=np.float64)
