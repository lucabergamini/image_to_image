from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import numpy

class DATASET(Dataset):

    def __init__(self, data_folder):
        #list of file names
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]
        self.len = len(self.data_names)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, item):
        """

        :param item: image index between 0-(len-1)
        :return: image
        """
        data = cv2.cvtColor(cv2.imread(self.data_names[item]), cv2.COLOR_BGR2RGB)
        if data.shape[0] != 256:
            data = cv2.resize(data,(256,256),interpolation=cv2.INTER_CUBIC)
        # random flip
        if numpy.random.randint(0,2,1)[0] == 1:
            data = cv2.flip(data,1)

        # CHANNEL FIRST
        data = self.transform(data)
        #data = data.transpose(2, 0, 1)

        return data


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    from matplotlib import pyplot
    dataloader_X = torch.utils.data.DataLoader(DATASET("/home/luca/Desktop/image_to_image/dataset/PHOTO/train"), batch_size=1,
                                               shuffle=False, num_workers=2)
    for i,d in enumerate(dataloader_X):

        pass