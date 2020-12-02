import numpy as np
import cv2
from augment.AugmentBase import AugmentBase


class HwTrans(AugmentBase):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha
        self.new_wh = (0, 0)
        self.old_wh = (0, 0)

    def trans_image(self, image):
        h, w = image.shape[:2]
        self.old_wh = (w, h)
        ratio = self._rand(-1 * self.alpha, self.alpha)
        new_ratio = h / w + ratio if (h / w + ratio) > 0 else h / w
        if ratio < 0:
            self.new_wh = (int(w), int(w * new_ratio))
        else:
            self.new_wh = (int(h / new_ratio), int(h))
        return cv2.resize(image, self.new_wh, cv2.INTER_CUBIC)

    def trans_label(self, label):
        return cv2.resize(label, self.new_wh, cv2.INTER_CUBIC)

    def trans_point(self, point):
        return np.divide(np.multiply(point, self.new_wh), self.old_wh).astype(np.int)

    def _check_param(self):
        if self.new_wh[0] == 0:
            raise Exception("Call function 'trans_image' first!")
