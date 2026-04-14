import torch
import torch.nn as nn

#https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8
#https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py -- this looks slightly incorrect in places

"""U-Net double convolutional block - 3x3 conv, normalize, ReLU"""
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.unet_conv_block = nn.Sequential(
                               nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU())

    def forward(self, x):
        return self.unet_conv_block(x)

"""single decoder block - upscales then double conv block"""
class UNetDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecodeBlock, self).__init__()
        self.decode_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=2, stride=2),
                UNetConvBlock(in_channels, out_channels)) #not in_channels//2 b/c it gets concatenated with equal dimensions from skip connection

    def forward(self, x, y):
        x = self.decode_block(x)
        x = torch.cat([x, y], dim=1) 
        return x

"""single encoder block - 2x pooling then double conv block"""
class UNetEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncodeBlock, self).__init__()
        self.unet_encode_block = nn.Sequential(
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 UNetConvBlock(in_channels=in_channels, out_channels=out_channels))

    def forward(self, x):
        return self.unet_encode_block(x)

"""residual block in standard u-net bottleneck."""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
                               nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(in_channels),
                               nn.ReLU(),
                               nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.residual_block(x)
        return self.relu(x + x0)

"""FFN for TransformerEncoder"""
class FFN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(512, 2048)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(2048, 512)

  def forward(self, x):
    x = self.linear(x)
    x = self.relu(x)
    return self.linear2(x)

"""Encoder from Attention is All You Need - used in core of generator bottleneck"""
class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.attention1_encoder = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.ffn_encoder = FFN()

        self.layer_norm = torch.nn.LayerNorm(512)

    def forward(self, x, mask):
        encoder_embedding_sublayer = self.attention1_encoder(x, x, x, mask)
        encoder_norm = self.layer_norm(x + encoder_embedding_sublayer)
        encoder_ffn = self.ffn_encoder(encoder_norm)
        encoder_output = self.layer_norm(encoder_norm + encoder_ffn)

        return encoder_output

"""put together the encode and decode blocks. This is 4 encode layers with 4 decode layers and 1 intermediate encode layer"""
class Generator(nn.Module):
    def __init__(self, channels, classes):
        super(Generator, self).__init__()
        self.channels = channels
        self.classes = classes

        #start model
        self.encode1    = UNetConvBlock(channels, 64)
        self.encode2    = UNetEncodeBlock(64, 128)
        self.encode3    = UNetEncodeBlock(128, 256)
        self.encode4    = UNetEncodeBlock(256, 512)
        self.encode5    = UNetEncodeBlock(512, 1024)
        self.encode6    = UNetEncodeBlock(1024, 2048)
        self.residual1  = ResidualBlock(2048)
        self.encoder    = TransformerEncoder()
        self.residual2  = ResidualBlock(2048)
        self.decode1    = UNetDecodeBlock(2048, 1024)
        self.decode2    = UNetDecodeBlock(1024, 512)
        self.decode3    = UNetDecodeBlock(512, 256)
        self.decode4    = UNetDecodeBlock(256, 128)
        self.decode5    = UNetDecodeBlock(128, 64)
        self.output     = nn.Conv2d(64, classes, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.encode1(x)
        x1 = self.encode2(x0)
        x2 = self.encode3(x1)
        x3 = self.encode4(x2)
        x4 = self.encode5(x3)
        x5 = self.encode6(x4)
        x5 = self.residual1(x5)
        x5 = self.encoder(x5)
        x5 = self.residual2(x5)
        x  = self.decode1(x5,x4)
        x  = self.decode2(x,x3)
        x  = self.decode3(x,x2)
        x  = self.decode4(x,x1)
        x  = self.decode5(x,x0)
        return self.output(x)
    
#Discriminator model
#https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
class Discriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential((nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=4, stride=2, padding=1)),
                                    nn.LeakyReLU(0.1))

        self.layer2 = nn.Sequential((nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1)),
                                    nn.BatchNorm2d(num_filters*2),
                                    nn.LeakyReLU(0.1))

        self.layer3 = nn.Sequential((nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1)),
                                    nn.BatchNorm2d(num_filters*4),
                                    nn.LeakyReLU(0.1))

        self.layer4 = nn.Sequential((nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1)),
                                    nn.BatchNorm2d(num_filters*8),
                                    nn.LeakyReLU(0.1))

        self.layer5 = (nn.Conv2d(num_filters*8, 1, kernel_size=4, stride=1, padding=1))


    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output