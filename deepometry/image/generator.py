import keras.preprocessing.image
import numpy
import skimage.exposure

import deepometry.image.iterator


class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def __init__(self,
                 gamma_adjust_range=0.0,
                 height_shift_range=0.0,
                 horizontal_flip=False,
                 preprocessing_function=None,
                 rotation_range=0.0,
                 vertical_flip=False,
                 width_shift_range=0.0):
        super(ImageDataGenerator, self).__init__(
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            preprocessing_function=preprocessing_function,
            rotation_range=rotation_range,
            vertical_flip=vertical_flip,
            width_shift_range=width_shift_range
        )

        self.gamma_adjust_range = gamma_adjust_range

    def flow(self, x,
             y=None,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix="",
             save_format="tif"):
        return deepometry.image.iterator.NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format
        )

    def flow_from_directory(self, directory,
                            target_size=(48, 48),
                            color_mode="rgb",
                            classes=None,
                            class_mode="categorical",
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix="",
                            save_format="tif",
                            follow_links=False):
        raise NotImplementedError()

    def random_transform(self, x, seed=None):
        if self.gamma_adjust_range:
            gamma = numpy.random.uniform(1.0 - self.gamma_adjust_range, 1.0 + self.gamma_adjust_range)
            gamma = max(gamma, 0.0)
            x = skimage.exposure.adjust_gamma(x, gamma=gamma)

        return super(ImageDataGenerator, self).random_transform(x, seed)
