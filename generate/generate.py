from torchvision import datasets
import numpy as np
import shutil

ROOT_FOLDER = '/data/tf-data'


def generate_like_mnist(loader, name):
    folder = ROOT_FOLDER + '/generate-' + name
    data_test = loader(folder, train=False, download=True)
    data_train = loader(folder, train=True, download=True)
    train = {'train_data': data_train.train_data, 'train_labels': data_train.train_labels}
    test = {'test_data': data_test.test_data, 'test_labels': data_test.test_labels}
    np.savez_compressed(ROOT_FOLDER + '/' + name + '.npz', **train, **test)
    shutil.rmtree(folder)


generate_like_mnist(datasets.MNIST, 'mnist')
generate_like_mnist(datasets.FashionMNIST, 'fashion-mnist')