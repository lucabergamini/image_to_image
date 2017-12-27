import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from networks.utils import AutoEncoder,PatchDiscriminator
from torchvision.transforms import Normalize



class AdversarialAutoEncoder(nn.Module):
    def __init__(self):
        super(AdversarialAutoEncoder, self).__init__()
        self.autoencoder = AutoEncoder(input_features=3)
        self.patch = PatchDiscriminator(input_features=3)

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    nn.init.xavier_normal(m.weight,1)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias,0)
                    #nn.init.normal(m.bias, 0,0.05)

    def forward(self,ten,batch_size=64):
        if self.training:
            #porto in range tanh
            ten = ten/127.5 -1
            ten_original = ten
            ten_reconstructed = self.autoencoder(ten)

            return ten_original,ten_reconstructed

        else:
            #ricostruisco
            #porto in range tanh
            ten = ten/127.5 -1
            ten_original = ten
            ten_reconstructed = self.autoencoder(ten)
            # uscita tanh
            return ten_original,ten_reconstructed

    @staticmethod
    def loss(original_images,reconstructed_images):
        """

        :param original_images: immagini originali
        :param reconstructed_images: immagini uscita autoencoder
        :param original_score: score immagini originali
        :param reconstructed_score: score immagini ricostruite
        :return:
        """
        #l1
        #porto in range 0-255
        #original_images = (original_images+1)*127.5
        #reconstructed_images = (reconstructed_images+1)*127.5
        l1 = nn.L1Loss()(reconstructed_images,original_images)

        return l1


class CycleGan(nn.Module):
    def __init__(self):
        super(CycleGan, self).__init__()
        #A
        self.autoencoder_G = AutoEncoder(input_features=3)
        self.patch_G = PatchDiscriminator(input_features=3)
        #B
        self.autoencoder_F = AutoEncoder(input_features=3)
        self.patch_F = PatchDiscriminator(input_features=3)

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    nn.init.normal(m.weight,0.0,0.02)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.normal(m.weight, 1.0, 0.02)
                nn.init.constant(m.bias, 0.0)


    def forward(self,ten_X,ten_Y):
        if self.training:
            #range
            #ten_X = Normalize((0.5,0.5,0,5),(0.5,0.5,0,5))(ten_X)
            #ten_X = ten_X / 127.5 - 1
            ten_original_X = ten_X
            ten_original_Y = ten_Y
            #forward A
            ten_X_Y = self.autoencoder_G(ten_X)
            #forward B
            ten_Y_X = self.autoencoder_F(ten_Y)
            #GAN LOSS
            ten_GAN_G = torch.cat((ten_original_Y,ten_X_Y),0)
            ten_GAN_F = torch.cat((ten_original_X,ten_Y_X),0)
            ten_classifications_G = self.patch_G(ten_GAN_F)
            ten_classifications_F = self.patch_F(ten_GAN_G)
            #CYCLE LOSS
            #self.eval()
            ten_Y_X_Y = self.autoencoder_G(ten_Y_X)
            ten_X_Y_X = self.autoencoder_F(ten_X_Y)
            #self.train()
            return ten_original_X,ten_original_Y,ten_classifications_G,ten_classifications_F,ten_X_Y_X,ten_Y_X_Y
            #return ten_original_X,ten_original_Y,ten_classifications_G,ten_classifications_F,ten_X_Y,ten_Y_X
        else:
            #range
            #ten_X = ten_X / 127.5 - 1
            ten_original_X = ten_X
            ten_original_Y = ten_Y
            #forward A
            ten_X_Y = self.autoencoder_G(ten_X)
            #forward B
            ten_Y_X = self.autoencoder_F(ten_Y)
            #double forward
            ten_Y_X_Y = self.autoencoder_G(ten_Y_X)
            ten_X_Y_X = self.autoencoder_F(ten_X_Y)
            return ten_original_X,ten_original_Y,ten_X_Y,ten_Y_X,ten_X_Y_X,ten_Y_X_Y
    @staticmethod
    def loss(ten_original_X,ten_original_Y,ten_classifications_G,ten_classifications_F,ten_X_Y_X,ten_Y_X_Y,batch_size):
        """

        :param ten_original_X:
        :param ten_original_Y:
        :param ten_classifications_G:
        :param ten_classifications_F:
        :param ten_X_Y_X:
        :param ten_Y_X_Y:
        :return:
        """
        #GAN loss
        gan_discriminator_G = nn.MSELoss()(ten_classifications_G[0:batch_size],torch.ones_like(ten_classifications_G[0:batch_size]).cuda())+\
                              nn.MSELoss()(ten_classifications_G[batch_size:],torch.zeros_like(ten_classifications_G[batch_size:]).cuda())

        gan_discriminator_F = nn.MSELoss()(ten_classifications_F[0:batch_size],
                                           torch.ones_like(ten_classifications_F[0:batch_size]).cuda()) + \
                              nn.MSELoss()(ten_classifications_F[batch_size:],
                                           torch.zeros_like(ten_classifications_G[batch_size:]).cuda())
        gan_discriminator_G /=2
        gan_discriminator_F /=2
        gan_autoencoder_G = nn.MSELoss()(ten_classifications_G[batch_size:],
                                           torch.ones_like(ten_classifications_G[batch_size:]).cuda())
        gan_autoencoder_F = nn.MSELoss()(ten_classifications_F[batch_size:],
                                         torch.ones_like(ten_classifications_F[batch_size:]).cuda())

        #CYCLE loss
        l1_G = nn.L1Loss()(ten_Y_X_Y,ten_original_Y)
        l1_F = nn.L1Loss()(ten_X_Y_X,ten_original_X)
        return l1_G,gan_autoencoder_G,gan_discriminator_G,l1_F,gan_autoencoder_F,gan_discriminator_F


if __name__ =="__main__":
    from torch.autograd import Variable
    import numpy
    a = Variable(torch.FloatTensor(numpy.random.randn(4,3,128,128)).cuda())
    x = PatchDiscriminator().cuda()
    x(a)