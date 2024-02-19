import torch
import torch.nn as nn


#double 3x3 convolution
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3,padding=1,padding_mode='reflect'),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1,padding_mode='reflect'),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace= True),
    )
    return conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.dwn_conv1 = dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trans1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride= 2)
        self.up_conv1 = dual_conv(1024,512)
        self.trans2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride= 2)
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride= 2)
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride= 2)
        self.up_conv4 = dual_conv(128,64)

        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):

        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x,x7], 1))

        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x,x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x,x3], 1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x,x1], 1))

        x = self.out(x)

        return x