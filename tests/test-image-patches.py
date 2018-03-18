import unittest
import numpy as np
import tensorflow as tf

from tf_data import ImagePatches


def _get_index_shape(shape):
    return tf.reshape(list(range(int(np.prod(shape)))), shape)


class _Tests(tf.test.TestCase):

    def _testIfPatchIsReversible(self, batch_count, patch_count, patch_dim, channels):
        original_size = np.multiply(patch_count, patch_dim).tolist()
        indexes = tf.tile(_get_index_shape([batch_count] + original_size + [1]), [1, 1, 1, channels])
        patches, _ = ImagePatches._split_into_patches(indexes, reversible=True, patch_dim=patch_dim)
        joined = ImagePatches._join_patches(patches, original_size, patch_dim=patch_dim, channels=channels)

        with tf.Session() as session:
            expected_patch_shape = [batch_count, np.prod(patch_count), np.prod(patch_dim) * channels]
            self.assertAllEqual(tf.shape(patches).eval(), expected_patch_shape)
            self.assertAllEqual(indexes.eval(), joined.eval())

    def testPatches(self):
        self._testIfPatchIsReversible(batch_count=1, patch_count=[2, 2], patch_dim=[2, 2], channels=1)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[2, 2], patch_dim=[2, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[2, 2], patch_dim=[2, 2], channels=3)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[1, 2], patch_dim=[2, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[5, 5], patch_dim=[1, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[5, 5], patch_dim=[1, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[5, 5], patch_dim=[1, 5], channels=2)
        self._testIfPatchIsReversible(batch_count=1, patch_count=[3, 3], patch_dim=[2, 2], channels=1)

        self._testIfPatchIsReversible(batch_count=5, patch_count=[2, 2], patch_dim=[2, 2], channels=1)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[2, 2], patch_dim=[2, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[2, 2], patch_dim=[2, 2], channels=3)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[1, 2], patch_dim=[2, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[5, 5], patch_dim=[1, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[5, 5], patch_dim=[1, 2], channels=2)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[5, 5], patch_dim=[1, 5], channels=2)
        self._testIfPatchIsReversible(batch_count=5, patch_count=[3, 3], patch_dim=[2, 2], channels=1)


if __name__ == '__main__':
    unittest.main()


