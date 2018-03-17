from torchvision import datasets
import numpy as np
import shutil

np.random.seed(0)
ROOT_FOLDER = '/data/tf-data'


def shuffle(prefix, data):
    count = len(getattr(data, prefix + '_data'))
    order = list(range(count))
    np.random.shuffle(order)
    return {
        prefix + '_data': getattr(data, prefix + '_data')[order],
        prefix + '_labels': getattr(data, prefix + '_labels')[order]
    }


def generate_like_mnist(loader, name):
    folder = ROOT_FOLDER + '/generate-' + name
    test = shuffle('test', loader(folder, train=False, download=True))
    train = shuffle('train', loader(folder, train=True, download=True))
    np.savez_compressed(ROOT_FOLDER + '/' + name + '.npz', **train, **test)
    shutil.rmtree(folder)


generate_like_mnist(datasets.MNIST, 'mnist')
generate_like_mnist(datasets.FashionMNIST, 'fashion-mnist')