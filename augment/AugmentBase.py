from abc import ABC, abstractmethod
import numpy as np


class AugmentBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def trans_image(self, image):
        pass

    @abstractmethod
    def trans_label(self, label):
        pass

    @abstractmethod
    def trans_point(self, point):
        pass

    @abstractmethod
    def _check_param(self):
        pass

    @staticmethod
    def _rand(a=0.0, b=1.0):  # 左开右闭
        return np.random.random() * (b - a) + a
