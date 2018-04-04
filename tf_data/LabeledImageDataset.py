import tensorflow as tf
import numpy as np
from tf_data.ImagePatches import ImagePatches


class LabeledImagePlaceholder:
    def __init__(self, info):
        self.info = info
        self._position = 0
        self.train_size = self.info.raw_data['train_data'].shape[0]
        self.image_flat = tf.placeholder(tf.uint8, [None, np.prod(self.info.dim_image)])
        self.image_byte = tf.placeholder(tf.float32, [None, np.prod(self.info.dim_image), 8])
        self.label = tf.placeholder(tf.int32, [None])
        self.image = tf.reshape(self.image_flat, [-1] + self.info.dim_image)
        self.image_float = tf.cast(self.image, tf.float32) / 255.0
        self.label_one_hot = tf.one_hot(self.label, self.info.label_count)
        self.patches = ImagePatches.build(self.image, self.info.width, self.info.height, self.info.color_channels)

    def _to_bytes(self, images):
        images_byte = np.reshape(images, (-1, np.prod(self.info.dim_image), 1))
        return np.unpackbits(images_byte, axis=2)

    def train(self, batch_size=40):
        start = self._position % self.train_size
        end = (self._position + batch_size) % self.train_size
        self._position = end

        def slice_with_range_restart(value):
            return value[start:end] if end > start else np.concatenate((value[start:], value[:end]))

        labels = slice_with_range_restart(self.info.raw_data['train_labels'])
        images = slice_with_range_restart(self.info.raw_data['train_data'])
        images = np.reshape(images, (-1, np.prod(self.info.dim_image)))

        return {
            self.image_flat: images,
            self.image_byte: self._to_bytes(images),
            self.label: labels,
        }

    def test(self, splits=1):
        labels = self.info.raw_data['test_labels']
        images = self.info.raw_data['test_data']
        images = np.reshape(images, (-1, np.prod(self.info.dim_image)))

        images_split = np.split(images, splits)
        labels_split = np.split(labels, splits)
        results = [{
            self.image_flat: images_split[i],
            self.image_byte: self._to_bytes(images_split[i]),
            self.label: labels_split[i]
        } for i in range(splits)]

        return results[0] if splits == 1 else results


class LabeledImageDataset:
    def __init__(self, width, height, color_channels, label_count, file):
        self.raw_file = file
        self.raw_data = np.load(file)
        self.dim_image = [width, height, color_channels]

        self.width = width
        self.height = height
        self.label_count = label_count
        self.image_values_count = int(np.prod(self.dim_image))
        self.color_channels = color_channels
        self.placeholder = lambda: LabeledImagePlaceholder(self)
