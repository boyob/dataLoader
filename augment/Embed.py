import numpy as np
import cv2
from PIL import Image
from augment.AugmentBase import AugmentBase


class Embed(AugmentBase):
    def __init__(self, output_hw=(300, 300), border_value=(128, 128, 128), alpha=0.15):
        super().__init__()
        self.output_hw = output_hw
        self.border_value = border_value
        self.alpha = alpha
        self.input_hw = (0, 0)
        self.ratio = 1  # input / output
        self.temp_hw = np.array([0, 0], dtype=np.int)
        self.dx_dy = (0, 0)

    def trans_image(self, image):
        self.input_hw = image.shape[:2]
        self.ratio = max(np.divide(image.shape[:2], self.output_hw))
        self.temp_hw = np.divide(image.shape[:2], self.ratio).astype(np.int)
        dx_dy1 = (int(self._rand(-1 * self.alpha * self.output_hw[1], self.alpha * self.output_hw[1])),
                  int(self._rand(-1 * self.alpha * self.output_hw[0], self.alpha * self.output_hw[0])))
        dx_dy2 = np.divide(np.subtract(self.output_hw, self.temp_hw), 2)[::-1]
        self.dx_dy = tuple(np.add(dx_dy1, dx_dy2).astype(np.int))
        return self._embed(image)

    def trans_label(self, label):
        self._check_param()
        return self._embed(label)

    def trans_point(self, point):
        self._check_param()
        point = np.divide(point, self.ratio).astype(np.int)
        point = np.sum((point, np.array(self.dx_dy)), axis=0)
        point[point < 0] = 0
        point[0:1][point[0:1] > self.output_hw[1]] = self.output_hw[1]
        point[1:2][point[1:2] > self.output_hw[0]] = self.output_hw[0]
        return np.ceil(point).astype(np.int32)

    def _check_param(self):
        if self.input_hw[0] == 0:
            raise Exception("Call function 'trans_image' first!")

    def _embed(self, im):
        im = cv2.resize(im, tuple(self.temp_hw[::-1]), cv2.INTER_CUBIC)
        empty = np.zeros(shape=(self.output_hw + (0,)), dtype=np.uint8)
        for pixel in self.border_value:
            channel = np.full(self.output_hw + (1,), pixel, dtype=np.uint8)
            empty = np.concatenate((empty, channel), axis=2)
        empty_pil = Image.fromarray(empty)
        im_pil = Image.fromarray((im))
        empty_pil.paste(im_pil, self.dx_dy)
        return np.array(empty_pil).astype(np.uint8)
