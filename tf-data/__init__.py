import drive
import os
import numpy as np


def mnist(path, file='mnist.npz'):
    drive.download(path, file, '18JJ5hzZC_pKF7dfpHHn2Ca2-7JZrHSqJ', 11507640)
    return np.load(os.path.join(path, file))


print(len(mnist('/data/tf-data')['train_labels']))