import torch
from torch.utils.data import Dataset
import numpy as np

class DASDataset(Dataset):
    def __init__(self, imagepath, labelpath, IDlist, chann, dim):
        super(DASDataset, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.IDlist = IDlist
        self.chann = chann
        self.dim = dim
    
    def __getitem__(self, index):
        image = np.fromfile(self.imagepath + self.IDlist[index], dtype=np.single)
        image_max = np.max(image)
        image_min = np.min(image)
        image -= image_min
        image /= (image_max - image_min)
        image = image.reshape(self.chann, self.dim[0], self.dim[1])
        label = np.fromfile(self.labelpath + self.IDlist[index], dtype=np.single)
        label = 2 * np.clip(label, 0, 1)
        label = label.reshape(self.chann, self.dim[0], self.dim[1])
        return image, label

    def __len__(self):
        return len(self.IDlist)