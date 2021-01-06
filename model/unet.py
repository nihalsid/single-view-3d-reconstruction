import torch
import torch.nn as nn


def get_conv_layer():
    return lambda in_c, out_c, k, p, s: nn.Conv2d(
        in_c, out_c, k, p, s
    )


def get_batchnorm_layer():
    return nn.BatchNorm2d


class Unet(nn.Module):
    """ The depth regressor for predicting the depth map given an rgb image.
    """

    def __init__(
        self,
        num_filters=32,
        channels_in=3,
        channels_out=3,
    ):
        super(Unet, self).__init__()

        conv_layer = get_conv_layer()

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = conv_layer(num_filters * 8, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv1 = conv_layer(num_filters * 8, num_filters * 8, 3, 1, 1)
        self.dconv2 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv3 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv4 = conv_layer(num_filters * 8 * 2, num_filters * 8, 3, 1, 1)
        self.dconv5 = conv_layer(num_filters * 8 * 2, num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = get_batchnorm_layer()

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)
        self.batch_norm8_0 = norm_layer(num_filters * 8)
        self.batch_norm8_1 = norm_layer(num_filters * 8)
        self.batch_norm8_2 = norm_layer(num_filters * 8)
        self.batch_norm8_3 = norm_layer(num_filters * 8)
        self.batch_norm8_4 = norm_layer(num_filters * 8)
        self.batch_norm8_5 = norm_layer(num_filters * 8)
        self.batch_norm8_6 = norm_layer(num_filters * 8)
        self.batch_norm8_7 = norm_layer(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
        # state size is (num_filters) x 128 x 128
        # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        # state size is (nc) x 256 x 256
        # output = self.tanh(d8)
        return d8


class UNetMini(nn.Module):
    """ A smaller UNet with half the convolution layers.
        Used for testing with the ground truth depth map where the input (240 x 320) and output (240 x 320) are not resized.
    """

    def __init__(
        self,
        num_filters=32,
        channels_in=3,
        channels_out=3,
    ):
        super(UNetMini, self).__init__()

        conv_layer = get_conv_layer()

        self.conv1 = conv_layer(channels_in, num_filters, 4, 2, 1)
        self.conv2 = conv_layer(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = conv_layer(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = conv_layer(num_filters * 4, num_filters * 8, 4, 2, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv5 = conv_layer(num_filters * 8 , num_filters * 4, 3, 1, 1)
        self.dconv6 = conv_layer(num_filters * 4 * 2, num_filters * 2, 3, 1, 1)
        self.dconv7 = conv_layer(num_filters * 2 * 2, num_filters, 3, 1, 1)
        self.dconv8 = conv_layer(num_filters * 2, channels_out, 3, 1, 1)

        norm_layer = get_batchnorm_layer()

        self.batch_norm = norm_layer(num_filters)
        self.batch_norm2_0 = norm_layer(num_filters * 2)
        self.batch_norm2_1 = norm_layer(num_filters * 2)
        self.batch_norm4_0 = norm_layer(num_filters * 4)
        self.batch_norm4_1 = norm_layer(num_filters * 4)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 240 x 320 
        e1 = self.conv1(input)
        # state size is (num_filters) x 120 x 160
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 60 x 80
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 30 x 40
        e4 = self.conv4(self.leaky_relu(e3))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 15 x 20
        d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(e4))))
        # state size is (num_filters x 4) x 30 x 40
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
        # state size is (num_filters x 2) x 60 x 80
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
        # state size is (num_filters) x 120 x 160
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.up(self.relu(d7)))
        # state size is (nc) x 240 x 320 
        # output = self.tanh(d8)
        return d8
