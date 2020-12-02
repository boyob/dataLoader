from AugmenterList import AugmenterList
import numpy as np
import torch


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, root, reader, augmenterList=None):
        # data_line = image_path x1,y1,x2,y2,...
        # data_item = [image_path, [[x1, y1], [x2, y2], ...]]
        self.data_items = []
        data_lines = open(root, 'r')
        for data_line in data_lines:
            line = data_line.strip().split(' ')
            image_path = line[0]
            image_boxes = np.array([int(coor) for coor in line[1].split(',')])
            image_boxes = np.array([[image_boxes[i], image_boxes[i + 1]] for i in range(0, len(image_boxes), 2)])
            self.data_items.append([image_path, image_boxes])
        self.augmentList = augmenterList
        self.loader = reader

    def __getitem__(self, item):
        image_path, image_boxes = self.data_items[item]
        image = self.loader(image_path)
        if self.augmentList is not None:
            image, image_boxes = self.augmentList.run(image, points=image_boxes)
        return image, image_boxes

    def __len__(self):
        return len(self.data_items)


if __name__ == '__main__':
    from reader import cv2_reader
    import cv2

    batch_size = 4
    reader = cv2_reader
    augmenterList = AugmenterList()

    train_data = MyDataSet(r'dataset\train.txt', reader, augmenterList)
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    for image_bs, points_bs in data_loader:
        for b in range(batch_size):
            image = np.array(image_bs[b])
            points = np.array(points_bs[b])
            for i in range(len(points) - 1):
                cv2.line(image, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 1, 4)
            cv2.line(image, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 1, 4)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(b) + '_20201201.jpg', image)
