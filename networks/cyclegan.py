import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from networks.utils import AutoEncoder, PatchDiscriminator
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
                    nn.init.xavier_normal(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0)
                    # nn.init.normal(m.bias, 0,0.05)

    def forward(self, ten, batch_size=64):
        if self.training:
            # porto in range tanh
            ten = ten / 127.5 - 1
            ten_original = ten
            ten_reconstructed = self.autoencoder(ten)

            return ten_original, ten_reconstructed

        else:
            # ricostruisco
            # porto in range tanh
            ten = ten / 127.5 - 1
            ten_original = ten
            ten_reconstructed = self.autoencoder(ten)
            # uscita tanh
            return ten_original, ten_reconstructed

    @staticmethod
    def loss(original_images, reconstructed_images):
        """

        :param original_images: immagini originali
        :param reconstructed_images: immagini uscita autoencoder
        :param original_score: score immagini originali
        :param reconstructed_score: score immagini ricostruite
        :return:
        """
        # l1
        # porto in range 0-255
        # original_images = (original_images+1)*127.5
        # reconstructed_images = (reconstructed_images+1)*127.5
        l1 = nn.L1Loss()(reconstructed_images, original_images)

        return l1


class CycleGan(nn.Module):
    def __init__(self, initial_fetures_map_enocer=32):
        super(CycleGan, self).__init__()
        # A
        self.autoencoder_G = AutoEncoder(input_features=3, initial_features_map=initial_fetures_map_enocer)
        self.patch_G = PatchDiscriminator(input_features=3)
        # B
        self.autoencoder_F = AutoEncoder(input_features=3, initial_features_map=initial_fetures_map_enocer)
        self.patch_F = PatchDiscriminator(input_features=3)

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    nn.init.normal(m.weight, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.normal(m.weight, 1.0, 0.02)
                nn.init.constant(m.bias, 0.0)

    def forward(self, ten_X, ten_Y, optimizer_encoder=None, optimizer_discriminator=None):
        if self.training:
            # range
            # ten_X = Normalize((0.5,0.5,0,5),(0.5,0.5,0,5))(ten_X)
            # ten_X = ten_X / 127.5 - 1
            ten_original_X = ten_X
            ten_original_Y = ten_Y
            # forward A
            ten_X_Y = self.autoencoder_G(ten_X)
            # forward B
            ten_Y_X = self.autoencoder_F(ten_Y)
            # CYCLE LOSS
            ten_Y_X_Y = self.autoencoder_G(ten_Y_X)
            ten_X_Y_X = self.autoencoder_F(ten_X_Y)
            # GAN ENCODER
            ten_X_Y_classification = self.patch_G(ten_X_Y)
            ten_Y_X_classification = self.patch_F(ten_Y_X)
            l1_G, loss_gan_autoencoder_G, l1_F, loss_gan_autoencoder_F = self.loss_encoder(ten_original_X, ten_original_Y,
                                                                                   ten_X_Y_classification,
                                                                                   ten_Y_X_classification, ten_X_Y_X,
                                                                                   ten_Y_X_Y)
            # BACKWARD ENCODER
            optimizer_encoder.zero_grad()
            (10 * (l1_G + l1_F) + loss_gan_autoencoder_G + loss_gan_autoencoder_F).backward()
            optimizer_encoder.step()
            # GAN DISCRIMINATOR
            ten_original_Y_classification = self.patch_G(ten_original_Y)
            ten_original_X_classification = self.patch_F(ten_original_X)
            ten_X_Y_classification = self.patch_G(ten_X_Y.detach())
            ten_Y_X_classification = self.patch_F(ten_Y_X.detach())
            loss_discriminator_G, loss_discriminator_F = self.loss_patch(ten_original_Y_classification,
                                                                         ten_X_Y_classification,
                                                                         ten_original_X_classification,
                                                                         ten_Y_X_classification)
            optimizer_discriminator.zero_grad()
            self.patch_G.zero_grad()
            loss_discriminator_G.backward()
            self.patch_F.zero_grad()
            loss_discriminator_F.backward()
            optimizer_discriminator.step()
            return ten_original_X, ten_original_Y, ten_original_Y_classification, ten_X_Y_classification, ten_original_X_classification, ten_Y_X_classification, ten_X_Y_X, ten_Y_X_Y, l1_G, loss_gan_autoencoder_G, loss_discriminator_G, l1_F, loss_gan_autoencoder_F, loss_discriminator_F
        else:
            # range
            # ten_X = ten_X / 127.5 - 1
            ten_original_X = ten_X
            ten_original_Y = ten_Y
            # forward A
            ten_X_Y = self.autoencoder_G(ten_X)
            # forward B
            ten_Y_X = self.autoencoder_F(ten_Y)
            # double forward
            ten_Y_X_Y = self.autoencoder_G(ten_Y_X)
            ten_X_Y_X = self.autoencoder_F(ten_X_Y)
            return ten_original_X, ten_original_Y, ten_X_Y, ten_Y_X, ten_X_Y_X, ten_Y_X_Y

    def loss_encoder(self, ten_original_X, ten_original_Y, ten_X_Y_classification, ten_Y_X_classification, ten_X_Y_X,
                     ten_Y_X_Y):
        gan_autoencoder_G = nn.MSELoss()(ten_X_Y_classification,
                                         torch.ones_like(ten_X_Y_classification).cuda())
        gan_autoencoder_F = nn.MSELoss()(ten_Y_X_classification,
                                         torch.ones_like(ten_Y_X_classification).cuda())

        # CYCLE loss
        l1_G = nn.L1Loss()(ten_Y_X_Y, ten_original_Y)
        l1_F = nn.L1Loss()(ten_X_Y_X, ten_original_X)
        return l1_G, gan_autoencoder_G, l1_F, gan_autoencoder_F

    def loss_patch(self, ten_original_Y_classification, ten_X_Y_classification, ten_original_X_classification,
                   ten_Y_X_classification):
        loss_discriminator_G = nn.MSELoss()(ten_original_Y_classification,
                                           torch.ones_like(ten_original_Y_classification).cuda()) + \
                              nn.MSELoss()(ten_X_Y_classification,
                                           torch.zeros_like(ten_X_Y_classification).cuda())

        loss_discriminator_F = nn.MSELoss()(ten_original_X_classification,
                                           torch.ones_like(ten_original_X_classification).cuda()) + \
                              nn.MSELoss()(ten_Y_X_classification,
                                           torch.zeros_like(ten_Y_X_classification).cuda())
        return loss_discriminator_G*0.5, loss_discriminator_F*0.5


