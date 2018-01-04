import os
import socket
from datetime import datetime
import torch

# just a class to store a rolling average
# useful to log to TB
class RollingMeasure(object):
    def __init__(self):
        self.measure = 0.0
        self.iter = 0

    def __call__(self, measure):
        # passo nuovo valore e ottengo average
        # se first call inizializzo
        if self.iter == 0:
            self.measure = measure
        else:
            self.measure = (1.0 / self.iter * measure) + (1 - 1.0 / self.iter) * self.measure
        self.iter += 1
        return self.measure


class ModelCheckpoint(object):
    def __init__(self,path,comment=""):
        self.log_dir = os.path.join(path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+"_"+comment)
        os.makedirs(self.log_dir)

    def __call__(self, net, comment=''):
        """
        :param net:
        :param comment:
        :return:
        """
        name = datetime.now().strftime('%b%d_%H-%M-%S') +  "_" + comment
        path = os.path.join(self.log_dir, name)
        torch.save(net.state_dict(), path)
