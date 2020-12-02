from PIL import Image
import os
import numpy as np
import cv2


def pil_reader(image_path):
    image = Image.open(image_path)
    # 如果是png图像，则去掉透明度通道。
    if os.path.splitext(image_path)[1] == '.png':
        image = image.convert('RGB')
    return np.array(image)


def cv2_reader(image_path):
    # 使用默认参数读取会忽略透明度通道。返回值类型是numpy.ndarray。
    image = cv2.imread(image_path)
    # 把默认BGR转为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
