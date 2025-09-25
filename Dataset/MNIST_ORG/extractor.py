import numpy as np
import struct
from PIL import Image

def read_idx(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Read the MNIST images
train_images = read_idx('t10k-images.idx3-ubyte')

# Save each image as PNG or JPG
for i in range(len(train_images)):
    img = Image.fromarray(train_images[i])
    img.save(f'images/mnist_image_{i}.png')  # Use '.jpg' instead of '.png' for JPG format
