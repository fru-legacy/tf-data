import drive
import os
import numpy as np


def mnist(path, file='mnist.npz'):
    drive.download(path, file, '18JJ5hzZC_pKF7dfpHHn2Ca2-7JZrHSqJ', 11507640)
    return np.load(os.path.join(path, file))


def fashion_mnist(path, file='fashion-mnist.npz'):
    drive.download(path, file, '1RptZ_WVaGvRTtn1Kf2UHWn0QkB5DIhJa', 30888555)
    return np.load(os.path.join(path, file))

#print(len(fashion_mnist('/data/tf-data')['train_labels']))
#print(fashion_mnist('/data/tf-data')['train_labels'][0:5])