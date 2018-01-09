import numpy
import cv2
import os
from  matplotlib import pyplot

FOLDER = "/home/luca/Desktop/CAT/dataset_raw"
DEST = "/home/luca/Desktop/CAT/dataset_256"


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
    data_names = [os.path.join(FOLDER,name) for name in sorted(os.listdir(FOLDER)) if ".cat" not in name]
    label_names = [os.path.join(FOLDER,name) for name in sorted(os.listdir(FOLDER)) if ".cat" in name]

    for i,(data_name,label_name) in enumerate(zip(data_names,label_names)):
        img = cv2.cvtColor(cv2.imread(data_name),cv2.COLOR_BGR2RGB)
        label = numpy.genfromtxt(label_name)
        #il primo e il numero di punti
        label = label[1:].astype("int")
        #devo essere sicuro siano 0
        label[label<0]  = 0
        label_x = label[::2]
        label_y = label[1::2]
        #mi servono gli estremi
        x_min,x_max = numpy.min(label_x),numpy.max(label_x)
        y_min,y_max = numpy.min(label_y),numpy.max(label_y)

        width = x_max-x_min
        height = y_max-y_min

        #la y va tenuta un po piu larga
        y_max += int(height/7)
        #devo avere le facce un po piu piccole (cioe al centro)
        x_min = max(0,x_min-int(width/9))
        x_max = min(img.shape[1],x_max+int(width/9))
        y_min = max(0, y_min - int(height / 9))
        y_max = min(img.shape[0], y_max + int(height / 9))

        img = _resize(img,(y_min,y_max,x_min,x_max))
        #print("a")
        data_name = data_name.split(os.sep)[-1]
        cv2.imwrite(os.path.join(DEST,data_name),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        #print("a")
        #pyplot.imshow(img)
        #pyplot.show()
        #a = 0
        print(i)

if __name__ == "__main__":
    pre_process(FOLDER)