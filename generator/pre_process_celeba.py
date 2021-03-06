import numpy
import cv2
import os
from  matplotlib import pyplot
FOLDER = "/home/luca/Desktop/CELEBA/img_align_celeba"
DEST = "/home/luca/Desktop/CELEBA/dataset_256"


def _resize(img,bbox):
    rescale_size = 256
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(rescale_size)
    sigma = numpy.sqrt(scale) / 2.0
    img = cv2.GaussianBlur(img,(3,3),sigma)
    img = cv2.resize(img,(rescale_size,rescale_size),interpolation=cv2.INTER_CUBIC)
    return img

def pre_process(FOLDER):
    data_names = [os.path.join(FOLDER,name) for name in sorted(os.listdir(FOLDER))][0:9993]
    bbox = (40, 218 - 30, 15, 178 - 15)
    for data_name in data_names:
        img = cv2.cvtColor(cv2.imread(data_name),cv2.COLOR_BGR2RGB)
        img = _resize(img,bbox)
        #print("a")
        data_name = data_name.split(os.sep)[-1]
        cv2.imwrite(os.path.join(DEST,data_name),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        #print("a")
        #pyplot.imshow(img)
        #pyplot.show()
        #a = 0

if __name__ == "__main__":
    pre_process(FOLDER)