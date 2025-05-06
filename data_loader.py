import os
import gzip
import numpy as np

def load_mnist(path='data/MNIST/raw'):
    """加载MNIST数据（自动处理压缩/未压缩文件）"""
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',
        'test_images': 't10k-images-idx3-ubyte',
        'test_labels': 't10k-labels-idx1-ubyte'
    }

    data = {}
    for key, filename in files.items():
        filepath = os.path.join(path, filename)
        gzpath = filepath + '.gz'

        # 优先使用解压后的文件
        if os.path.exists(filepath):
            print(f"Loading {filename} from {filepath}")
            with open(filepath, 'rb') as f:
                if 'images' in key:
                    data[key] = parse_idx3(f)
                else:
                    data[key] = parse_idx1(f)
        elif os.path.exists(gzpath):
            print(f"Loading {filename} from {gzpath}")
            with gzip.open(gzpath, 'rb') as f:
                if 'images' in key:
                    data[key] = parse_idx3(f)
                else:
                    data[key] = parse_idx1(f)
        else:
            raise FileNotFoundError(f"找不到MNIST文件: {filename}[.gz]")

    # 归一化并reshape图像数据
    data['train_images'] = data['train_images'].reshape(-1, 1, 28, 28) / 255.0
    data['test_images'] = data['test_images'].reshape(-1, 1, 28, 28) / 255.0

    return (data['train_images'], data['train_labels'],
            data['test_images'], data['test_labels'])

def parse_idx3(f):
    """解析idx3-ubyte格式"""
    magic = np.frombuffer(f.read(4), dtype='>i4')[0]
    if magic != 2051: raise ValueError("非法的MNIST图像文件")
    dims = np.frombuffer(f.read(12), dtype='>i4')
    return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)

def parse_idx1(f):
    """解析idx1-ubyte格式"""
    magic = np.frombuffer(f.read(4), dtype='>i4')[0]
    if magic != 2049: raise ValueError("非法的MNIST标签文件")
    count = np.frombuffer(f.read(4), dtype='>i4')[0]
    return np.frombuffer(f.read(count), dtype=np.uint8)

def get_data_loaders():
    try:
        return load_mnist()
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        print("请确保：")
        print("1. data/MNIST/raw/目录包含正确的MNIST文件")
        print("2. 文件未被其他程序占用")
        raise