import urllib.request
import gzip
import os
import struct
import numpy as np

# URLs for the MNIST dataset
urls = {
    'train_img': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_lbl': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_img':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_lbl':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
}

data_dir = 'Data'

def download_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for name, url in urls.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                # Fallback to mirror if needed, or user can manually place files
                raise

def load_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_data():
    download_data()
    
    print("Loading MNIST data...")
    X_train = load_idx(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    X_test  = load_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test  = load_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    print(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    load_data()
