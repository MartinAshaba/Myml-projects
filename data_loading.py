import numpy as np
import struct

# Specify the file path
#file_path = "path/to/t10k-images.idx3-ubyte"
# Replace with the actual file path
#file_path ="This PC/Documents/DDSAI/data_sets/data/t10k-images.idx3-ubyte"
#file_path = "C:\\Documents\\DDSAI\\data_sets\\data\\t10k-images.idx3-ubyte"  # Windows path
#file_path = "C:\Users\ASHAMARTS\Documents\DDSAI\data_sets\data\t10k-images.idx3-ubyte" syntax error

file_path = r"C:\Users\ASHAMARTS\Documents\DDSAI\data_sets\data\t10k-images.idx3-ubyte"




# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Read the header information (magic number, number of images, image dimensions)
    magic_number = struct.unpack('>I', file.read(4))[0]
    num_images = struct.unpack('>I', file.read(4))[0]
    num_rows = struct.unpack('>I', file.read(4))[0]
    num_cols = struct.unpack('>I', file.read(4))[0]

    # Read and load the image data
    image_data = np.fromfile(file, dtype=np.uint8)

# Reshape the image data to match the dimensions
image_data = image_data.reshape(num_images, num_rows, num_cols)
