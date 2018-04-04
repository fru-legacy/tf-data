import tf_data.drive
import os
from tf_data.LabeledImageDataset import LabeledImageDataset


def mnist(path, file='mnist.npz'):
    drive.download(path, file, '18JJ5hzZC_pKF7dfpHHn2Ca2-7JZrHSqJ', 11507640)
    return LabeledImageDataset(28, 28, 1, 10, os.path.join(path, file))


def fashion_mnist(path, file='fashion-mnist.npz'):
    drive.download(path, file, '1RptZ_WVaGvRTtn1Kf2UHWn0QkB5DIhJa', 30888555)
    return LabeledImageDataset(28, 28, 1, 10, os.path.join(path, file))

# Usage example:
# data = fashion_mnist('/data/tf_data').placeholder()
# print(data.test(splits=2)[0][data.image_flat].shape)
# print(data.train()[data.image_flat].shape)
