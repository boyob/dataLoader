import numpy as np
import cv2
from augment.AugmentBase import AugmentBase


class Perspective(AugmentBase):
    def __init__(self, border_value, alpha=0.1):
        super().__init__()
        self.borderValue = border_value  # (R, G, B)
        self.alpha = alpha
        self.M = np.zeros((3, 3))

    def trans_image(self, image):
        h, w = image.shape[:2]
        pts1 = np.float32([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])
        pts2 = np.float32([(0 + self._rand_trans(w), 0 + self._rand_trans(h)),
                           (w - 1 + self._rand_trans(w), 0 + self._rand_trans(h)),
                           (w - 1 + self._rand_trans(w), h - 1 + self._rand_trans(h)),
                           (0 + self._rand_trans(w), h - 1 + self._rand_trans(h))])
        # 得到左上边界坐标。
        min2x, min2y, _, _ = self._extreme_value(pts2)
        # 得到平移后的坐标，平移后左上边界在像素坐标轴上。
        pts2 = self._translate(pts2)
        # 确定输出尺寸和变换矩阵
        _, _, self.max2x, self.max2y = self._extreme_value(pts2)
        self.M = cv2.getPerspectiveTransform(pts1, pts2)
        res = cv2.warpPerspective(image, self.M, (int(self.max2x), int(self.max2y)), borderValue=self.borderValue)
        return res

    def trans_label(self, label):
        self._check_param()
        res = cv2.warpPerspective(label, self.M, (int(self.max2x), int(self.max2y)), borderValue=self.borderValue)
        return res

    def trans_point(self, point):
        self._check_param()
        point = np.append(point, np.array([1]))
        p = np.sum(self.M * point, axis=1)
        p = np.round(np.divide(p, p[2])).astype(np.int)[:2]
        return p

    def _check_param(self):
        if self.M[2][2] == 0:
            raise Exception("Call function 'trans_image' first!")

    @staticmethod
    def _extreme_value(a):
        minx, miny, maxx, maxy = min(a[:, 0]), min(a[:, 1]), max(a[:, 0]), max(a[:, 1])
        return minx, miny, maxx, maxy

    def _translate(self, points):
        minx, miny, _, _ = self._extreme_value(points)
        return np.subtract(points, np.array([minx, miny]))

    def _rand_trans(self, r):
        return self._rand(-1 * self.alpha * r, self.alpha * r)
