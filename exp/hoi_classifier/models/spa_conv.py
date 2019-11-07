import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

def bbox_trans(human_box_roi, object_box_roi, size=64):
    human_box = human_box_roi.copy()
    object_box = object_box_roi.copy()

    union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = union_box[3] - union_box[1] + 1
    width = union_box[2] - union_box[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)
    human_box[0] -= union_box[0]
    human_box[2] -= union_box[0]
    human_box[1] -= union_box[1]
    human_box[3] -= union_box[1]
    object_box[0] -= union_box[0]
    object_box[2] -= union_box[0]
    object_box[1] -= union_box[1]
    object_box[3] -= union_box[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (union_box[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (union_box[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def gen_spatial_map(human_box, object_box):
    hbox, obox = bbox_trans(human_box, object_box)
    spa_map = np.zeros((2, 64, 64), dtype='float32')
    spa_map[0, int(hbox[1]):int(hbox[3]) + 1, int(hbox[0]):int(hbox[2]) + 1] = 1
    spa_map[1, int(obox[1]):int(obox[3]) + 1, int(obox[0]):int(obox[2]) + 1] = 1
    return spa_map


class SpaConv(nn.Module):
    def __init__(self):
        super(SpaConv, self).__init__()
        # (batch,64,64,2)->(batch,60,60,64)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5)
        # (batch,60,60,64)->(batch,30,30,64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (batch,30,30,64)->(batch,26,26,32)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        # (batch,26,26,32)->(batch,13,13,32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.hidden = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(5408, 1024))

    def forward(self, spa_map):
        conv1 = self.conv1(spa_map)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        pool_feat = pool2.view(spa_map.shape[0], -1)
        return self.hidden(pool_feat)


class VerbGivenSpaConv(nn.Module):
    def __init__(self):
        pass

    def forward(self, hboxes, oboxes):
        spa_maps = np.zeros((len(hboxes), 2, 64, 64))
        for i in range(len(hboxes)):
            spa_maps[i] = gen_spatial_map(hboxes[i, :], oboxes[i, :])

