import tensorflow as tf
import numpy as np


def split_into_patches(images, patch_size=[2, 2], channels=1, reversible=False):
    kernel = [1] + patch_size + [1]
    stride = kernel if reversible else [1, 1, 1, 1]
    result = tf.extract_image_patches(images, kernel, strides=stride, rates=[1, 1, 1, 1], padding='VALID')

    size = int(result.get_shape()[3])
    count = int(np.prod(result.get_shape()[1:3]))

    return tf.reshape(result, [-1, size]), size, count


def join_patches(patches, original_size, patch_size=[2, 2], channels=1):

    def implementation(patches_numpy):
        patch_fragments = np.reshape(patches_numpy, [-1, patch_size[1] * channels])
        patch_count = np.divide(original_size, patch_size).astype(int).tolist()
        batch_count = int(np.prod(patch_fragments.shape) / (np.prod(original_size) * channels))
        return np.array([
            patch_fragments[y + x*patch_size[0] + section*patch_size[0]*patch_count[1]]
            # Thirdly go to next section of patches
            for section in range(patch_count[0]*batch_count)
            # Secondly move to the next line
            for y in range(patch_size[0])
            # First iterate through all patches on the current line
            for x in range(patch_count[1])
        ])

    result = tf.py_func(implementation, [patches], patches.dtype)
    return tf.reshape(result, shape=[-1] + original_size + [channels])
