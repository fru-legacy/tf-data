import tensorflow as tf
import numpy as np


def _split_into_patches(images, patch_dim, reversible=False):
    kernel = [1] + patch_dim + [1]
    stride = kernel if reversible else [1, 1, 1, 1]
    result = tf.extract_image_patches(images, kernel, strides=stride, rates=[1, 1, 1, 1], padding='VALID')
    count = int(np.prod(result.get_shape()[1:3]))
    size = int(result.get_shape()[3])
    return tf.reshape(result, [-1, count, size]), count


def _join_patches(patches, original_size, patch_dim, channels):

    def implementation(patches_numpy):
        patch_fragments = np.reshape(patches_numpy, [-1, patch_dim[1] * channels])
        patch_count = np.divide(original_size, patch_dim).astype(int).tolist()
        batch_count = int(np.prod(patch_fragments.shape) / (np.prod(original_size) * channels))
        return np.array([
            patch_fragments[y + x*patch_dim[0] + section*patch_dim[0]*patch_count[1]]
            # Thirdly go to next section of patches
            for section in range(patch_count[0]*batch_count)
            # Secondly move to the next line
            for y in range(patch_dim[0])
            # First iterate through all patches on the current line
            for x in range(patch_count[1])
        ])

    result = tf.py_func(implementation, [patches], patches.dtype)
    return tf.reshape(result, shape=[-1] + original_size + [channels])


def _integrate_patches_in_batch(data):
    assert len(data.get_shape()) == 3
    return tf.reshape(data, [-1, int(data.get_shape()[-1])])


def _extract_patches_from_batch(data, all_count):
    assert len(data.get_shape()) == 2
    return tf.reshape(data, [-1, all_count, int(data.get_shape()[-1])])


def _compute_patch_size_factors(patch_size):
    patch_sum = int(np.prod(patch_size))
    return [x for x in range(patch_sum) if patch_sum % x == 0]


class ImagePatches:
    @staticmethod
    def build(image, width, height, color_channels):
        def builder(patch_dim, auxiliary_max_count=None):
            return ImagePatches(image, width, height, color_channels, patch_dim, auxiliary_max_count)
        return builder

    def __init__(self, image, width, height, color_channels, patch_dim, auxiliary_max_count):
        self._patch_dim = patch_dim
        self._image_size = [height, width]
        self._color_channels = color_channels

        reversible, self._reversible_count = _split_into_patches(image, patch_dim, reversible=True)
        auxiliary, auxiliary_count = _split_into_patches(image, patch_dim, reversible=False)

        if auxiliary_max_count and auxiliary_max_count < auxiliary_count:
            all_indices = tf.random_shuffle(list(range(auxiliary_count)))
            indices = tf.slice(all_indices, [0], [auxiliary_max_count])
            auxiliary_count = auxiliary_max_count
            auxiliary = tf.gather(auxiliary, indices, axis=1)

        self.count = int(self._reversible_count + auxiliary_count)
        self.data = _integrate_patches_in_batch(tf.concat([reversible, auxiliary], 1))
        self.size = int(np.prod(patch_dim))

    def restored_image_summary(self, name, generated, max_outputs=3):
        generated = _extract_patches_from_batch(generated, self.count)
        assert generated.get_shape()[2] == self.size
        generated = tf.slice(generated, [0, 0, 0], [-1, self._reversible_count, -1])
        restored = _join_patches(generated, self._image_size, self._patch_dim, self._color_channels)
        return tf.summary.image(name, restored, max_outputs)
