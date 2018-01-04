import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy


class ResidualBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super(ResidualBlock, self).__init__()
        self.ref_1 = nn.ReflectionPad2d(padding=(1, 1, 1, 1))
        self.ref_2 = nn.ReflectionPad2d(padding=(1, 1, 1, 1))
        self.conv_1 = nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=3, padding=0,
                                stride=1)
        self.conv_2 = nn.Conv2d(in_channels=output_features, out_channels=output_features, kernel_size=3, padding=0,
                                stride=1)
        #self.conv_skip = nn.Conv2d(in_channels=input_features,out_channels=output_features,kernel_size=1,padding=0,stride=1)

        self.bn_1 = nn.InstanceNorm2d(num_features=output_features, affine=False)
        self.bn_2 = nn.InstanceNorm2d(num_features=output_features, affine=False)

    def forward(self, ten):
        # skip
        ten_init = ten
        #ten_init = self.conv_skip(ten_init)
        # flow
        ten = self.ref_1(ten)
        ten = self.conv_1(ten)
        ten = self.bn_1(ten)
        ten = F.relu(ten, True)
        ten = self.ref_2(ten)
        ten = self.conv_2(ten)
        ten = self.bn_2(ten)
        ten = ten + ten_init
        return ten

    def __call__(self, *args, **kwargs):
        return super(ResidualBlock, self).__call__(*args, **kwargs)


class AutoEncoder(nn.Module):
    def __init__(self, input_features,initial_features_map=32):
        super(AutoEncoder, self).__init__()
        self.c7s1_32 = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=input_features, out_channels=initial_features_map, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(num_features=initial_features_map, affine=False),
            nn.ReLU(True)
        )
        d64 = nn.Sequential(
            nn.Conv2d(in_channels=initial_features_map, out_channels=initial_features_map*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=initial_features_map*2, affine=False),
            nn.ReLU(True)
        )
        d128 = nn.Sequential(
            nn.Conv2d(in_channels=initial_features_map*2, out_channels=initial_features_map*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=initial_features_map*4, affine=False),
            nn.ReLU(True)
        )
        self.down = nn.Sequential(
            d64,
            d128
        )
        self.residuals = nn.ModuleList()
        for i in range(9):
            self.residuals.append(ResidualBlock(input_features=initial_features_map*4, output_features=initial_features_map*4))

        u64 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=initial_features_map*4, out_channels=initial_features_map*2, kernel_size=3, padding=1,
                               stride=2, output_padding=1),
            nn.InstanceNorm2d(num_features=initial_features_map*2, affine=False),
            nn.ReLU(True)
        )
        u32 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=initial_features_map*2, out_channels=initial_features_map, kernel_size=3, padding=1,
                               stride=2, output_padding=1),
            nn.InstanceNorm2d(num_features=initial_features_map, affine=False),
            nn.ReLU(True)
        )
        self.up = nn.Sequential(
            u64,
            u32
        )
        self.c7s1_3 = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=initial_features_map, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, ten):
        # predown
        ten = self.c7s1_32(ten)
        # down
        ten = self.down(ten)
        # residuals
        for r in self.residuals:
            ten = r(ten)
        # up
        ten = self.up(ten)
        # postup
        ten = self.c7s1_3(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(AutoEncoder, self).__call__(*args, **kwargs)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_features):
        super(PatchDiscriminator, self).__init__()

        self.c64 = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.c128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        self.c256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.c512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=512, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, ten,features=False):
        if features:
            ten = self.c64(ten)
            ten_128 = self.c128(ten)
            ten = self.c256(ten_128)
            ten = self.c512(ten)
            ten = self.c1(ten)
            return ten,ten_128

        else:
            ten = self.c64(ten)
            ten = self.c128(ten)
            ten = self.c256(ten)
            ten = self.c512(ten)
            ten = self.c1(ten)
            return ten

    def __call__(self, *args, **kwargs):
        return super(PatchDiscriminator, self).__call__(*args, **kwargs)


class ImagesPool(object):
    def __init__(self,pool_size):
        self.pool_size = pool_size
        self.images = []
        self.image = None

    def add(self,image):
        if len(self.images) < self.pool_size:
            self.images.append(image)
            self.image = image
        else:
            p = numpy.random.rand(1)[0]
            if p > 0.5:
                id = numpy.random.randint(0,self.pool_size,1)[0]
                tmp = self.images[id].clone()
                self.images[id] = image
                image = tmp
            self.image = image

    def __call__(self, *args, **kwargs):
        return self.image

if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy

    a = Variable(torch.FloatTensor(numpy.random.randn(4, 3, 32, 32)).cuda())
    x = ResidualBlock(input_features=3, output_features=6).cuda()
    x(a)
