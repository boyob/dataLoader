from Config import dataAugment
from augment.HsvTrans import HsvTrans
from augment.LrFlip import LrFlip
from augment.HwTrans import HwTrans
from augment.Perspective import Perspective
from augment.Embed import Embed
import numpy as np


class AugmenterList:
    def __init__(self):
        hsvTrans = HsvTrans(dataAugment['hsv'])
        lrFlip = LrFlip()
        hwTrans = HwTrans(dataAugment['hwTrans_alpha'])
        perspective = Perspective(dataAugment['perspective_border_value'], dataAugment['perspective_alpha'])
        embed = Embed(dataAugment['output_hw'], dataAugment['embed_border_value'], dataAugment['embed_alpha'])
        self.augmentList = [hsvTrans, lrFlip, hwTrans, perspective, embed]

    def run(self, image, label=None, points=None):
        for augment in self.augmentList:
            image = augment.trans_image(image)
        pts = []
        for point in points:
            for augment in self.augmentList[1:]:
                point = augment.trans_point(point)
            pts.append(point)
        return image, np.array(pts)
