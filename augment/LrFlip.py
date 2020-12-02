import cv2
import numpy as np
from augment.AugmentBase import AugmentBase


class LrFlip(AugmentBase):
    def __init__(self):
        super().__init__()
        self.hw = (0, 0)
        self.flip = False

    def trans_image(self, image):
        self.hw = image.shape[:2]
        self.flip = True if self._rand() < 0.5 else False
        if self.flip:
            return cv2.flip(image, 1)
        else:
            return image

    def trans_label(self, label):
        self._check_param()
        if self.flip:
            return cv2.flip(label, 1)
        else:
            return label

    def trans_point(self, point):
        self._check_param()
        if self.flip:
            point[0] = np.subtract(self.hw[1], point[0])
        return point

    def _check_param(self):
        if self.hw[0] == 0:
            raise Exception("Call function 'trans_image' first!")
