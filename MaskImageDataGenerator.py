import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class MaskImageDataGenerator(object):

    def __init__(self, rot, zoom, hoz_flip, ver_flip, wsr, hsr, sr, br, fill_mode="constant"):
        self.dgen = ImageDataGenerator(rotation_range=rot,
                                       zoom_range=zoom,
                                       horizontal_flip=hoz_flip,
                                       vertical_flip=ver_flip,
                                       width_shift_range=wsr,
                                       height_shift_range=hsr,
                                       shear_range=sr,
                                       brightness_range=br,
                                       fill_mode=fill_mode)

    def flow(self, x, y):
        dataset = self.dgen.flow(x, y)
        return dataset