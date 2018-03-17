import unittest
import numpy as np
import tensorflow as tf

from tf_data import ImagePatches


def _get_index_shape(shape):
    return tf.reshape(list(range(int(np.prod(shape)))), shape)


class _Tests(tf.test.TestCase):

    def _testIfPatchIsReversible(self, patch_count, patch_size, channels):
        original_size = np.multiply(patch_count, patch_size).tolist()
        indexes = tf.tile(_get_index_shape([1] + original_size + [1]), [1, 1, 1, channels])
        patches, _, _ = ImagePatches._split_into_patches(indexes, reversible=True, patch_size=patch_size, channels=channels)
        joined = ImagePatches._join_patches(patches, original_size, patch_size=patch_size, channels=channels)

        with tf.Session() as session:
            print(joined.eval())
            self.assertAllEqual(tf.shape(patches).eval(), [np.prod(patch_count), np.prod(patch_size) * channels])
            self.assertAllEqual(indexes.eval(), joined.eval())

    def testPatches(self):
        self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=1)
        self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=2)
        self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=3)
        self._testIfPatchIsReversible(patch_count=[1, 2], patch_size=[2, 2], channels=2)
        self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 2], channels=2)
        self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 2], channels=2)
        self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 5], channels=2)
        self._testIfPatchIsReversible(patch_count=[3, 3], patch_size=[2, 2], channels=1)


if __name__ == '__main__':
    unittest.main()


