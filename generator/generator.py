from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
class DATASET(Dataset):
    def __init__(self, data_folder):
        #list of file names
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]
        self.len = len(self.data_names)
        transform_list = [transforms.Resize((256,256),Image.BICUBIC),
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
        return data


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    from matplotlib import pyplot
    dataloader_X = torch.utils.data.DataLoader(DATASET("/home/lapis/Desktop/image_to_image/dataset/PHOTO/train"), batch_size=1,
                                               shuffle=False, num_workers=1)
    for i,d in enumerate(dataloader_X):

        pass