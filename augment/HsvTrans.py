import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from augment.AugmentBase import AugmentBase


class HsvTrans(AugmentBase):
    def __init__(self, hsv=(0.1, 1.1, 1.1)):
        super().__init__()
        self.hsv_range = np.array(hsv, dtype=np.float32)
        self.hsv_delta = np.array([0, 1, 1], dtype=np.float32)

    def trans_image(self, image):
        # 1. 根据hsv_range生成hsv_delta
        self.hsv_delta[0] = self._rand(-1 * self.hsv_range[0], self.hsv_range[0])
        random_s = self._rand(1, self.hsv_range[1])
        self.hsv_delta[1] = random_s if self._rand() < 0.5 else 1 / random_s
        random_v = self._rand(1, self.hsv_range[2])
        self.hsv_delta[2] = random_v if self._rand() < 0.5 else 1 / random_v
        # 2.根据hsv_delta变换
        return self._hsv(image)

    def trans_label(self, label):
        self._check_param()
        return self._hsv(label)

    def trans_point(self, point):
        return point

    def _check_param(self):
        if not np.subtract(self.hsv_delta, (0, 1, 1)).any() != 0:
            raise Exception("Call function 'trans_image' first!")

    def _hsv(self, im):
        hsv_im = rgb_to_hsv(np.divide(im, 255))
        hsv_im[..., 0] += self.hsv_delta[0]
        hsv_im[..., 0][hsv_im[..., 0] > 1] -= 1
        hsv_im[..., 0][hsv_im[..., 0] < 0] += 1
        hsv_im[..., 1] *= self.hsv_delta[1]
        hsv_im[..., 2] *= self.hsv_delta[2]
        hsv_im[hsv_im > 1] = 1
        hsv_im[hsv_im < 0] = 0
        im = np.multiply(hsv_to_rgb(hsv_im), 255).astype(np.uint8)
        return im
