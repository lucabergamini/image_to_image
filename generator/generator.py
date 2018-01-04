from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import torch
import numpy

class DATASET(Dataset):
    def __init__(self, data_folder):
        #list of file names
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]
        self.len = len(self.data_names)
        transform_list = [transforms.Resize((286,286),Image.BICUBIC),
                          transforms.RandomCrop(256),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
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
        data = Image.open(self.data_names[item])

        # CHANNEL FIRST
        data = self.transform(data)
        if data.shape[0] == 1:
            data = torch.cat((data,)*3,dim=0)
        return data

class DATASET_SLURM(Dataset):
    def __init__(self, data_folder):
        #expected a file named data
        # open the file
        self.file = open(os.path.join(data_folder, "data"), "rb")
        #get size
        self.size  = numpy.fromfile(self.file, dtype=numpy.uint32, count=1)[0]
        # get len
        self.len = int((os.path.getsize(os.path.join(data_folder, "data"))-4) / (self.size * self.size * 3))
        #trasform
        transform_list = [transforms.Resize((286,286),Image.BICUBIC),
                          transforms.RandomCrop(256),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))
                          ]

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
        offset = 4+item * 3 * self.size * self.size
        self.file.seek(offset)
        data = numpy.fromfile(self.file, dtype=numpy.uint8, count=(3 * self.size * self.size)).reshape(self.size,self.size,3)
        data = Image.fromarray(data)
        # trasform
        data = self.transform(data)
        return data



if __name__ == "__main__":
    from torchvision.utils import make_grid
    from matplotlib import pyplot
    dataloader_X = torch.utils.data.DataLoader(DATASET_SLURM("/home/luca/Desktop/dataset_to_slurm/"), batch_size=1,
                                               shuffle=False, num_workers=1)
    for i,d in enumerate(dataloader_X):
        d = d[0]
        d = numpy.transpose(d.numpy(),(1,2,0))
        pyplot.imshow(d)
        pyplot.show()
        pass