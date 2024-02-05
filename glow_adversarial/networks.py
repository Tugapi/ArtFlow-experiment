import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from glow_adversarial.glow_model import Glow


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style_std, style_mean):
        assert (content.size()[:2] == style_std.size()[:2]) and (content.size()[:2] == style_mean.size()[:2])
        size = content.size()
        content_mean, content_std = self.calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        n, c = size[:2]
        feat_var = feat.view(n, c, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(n, c, 1, 1)
        feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
        return feat_mean, feat_std


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)
        return x * tmp1


class Blur2d(nn.Module):
    """
        Low-pass filter applied when resampling activations
    """

    def __init__(self, f=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2) - 1) / 2),
                groups=x.size(1)
            )
            return x
        else:
            return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2 ** 0.5,  # gain for He init
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            One FC layer of the mapping network
        """
        super().__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, input):
        if self.bias is not None:
            out = F.linear(input, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(input, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class MappingNetwork(nn.Module):
    def __init__(self,
                 latent_z_length=512,
                 w_length=512,
                 normalize=True,
                 use_wscale=True,  # Enable equalized learning rate?
                 lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 gain=2 ** 0.5  # gain for He init
                 ):
        super().__init__()
        self.mapping_fmaps = latent_z_length
        self.net = nn.Sequential(
            FC(self.mapping_fmaps, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(w_length, w_length, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize = normalize
        self.pixel_norm = PixelNorm()

    def forward(self, input):
        if self.normalize:
            input = self.pixel_norm(input)
        out = self.net(input)
        return out


class AffineA(nn.Module):
    def __init__(self, w_length, z_channel, use_wscale=False):
        super().__init__()
        self.z_channel = z_channel
        self.affine_transform = FC(w_length, z_channel * 2, use_wscale=use_wscale)

    def forward(self, w):
        style = self.affine_transform(w)
        scale = style[:, :self.z_channel]
        bias = style[:, self.z_channel:]

        return scale, bias


class NetG(nn.Module):
    def __init__(self,
                 in_resolution=128,
                 in_channel=3, n_flow=4, n_block=2, affine=True, conv_lu=True, use_sigmoid=True, latent_z_length=512,
                 w_length=512, normalize=True, use_wscale=True, lrmul=0.01, gain=2 ** 0.5):
        super().__init__()
        self.adaIN = AdaIN()
        self.glow = Glow(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu, use_sigmoid=use_sigmoid)

        self.z_shapes = self.calc_z_shapes(in_channel, in_resolution, n_block)
        self.w_length = w_length

        self.use_wscale = use_wscale
        self.mapping_network = MappingNetwork(latent_z_length, w_length, normalize, use_wscale, lrmul, gain)

        self.affineAs = nn.ModuleList(AffineA(self.w_length, z_shape[0], self.use_wscale) for z_shape in self.z_shapes)

    def forward(self, input, latent_z):
        w = self.mapping_network(latent_z)

        # TODO: apply truncation trick
        z_outs = self.glow(input)  # list of z_outs with different sizes
        z_modifys = []
        for i, z_out in enumerate(z_outs):
            scale, bias = self.affineAs[i](w)
            scale = scale.unsqueeze(2).unsqueeze(3)
            bias = bias.unsqueeze(2).unsqueeze(3)
            z_modify = self.adaIN(z_out, scale, bias)
            z_modifys.append(z_modify)

        return self.glow.reverse(z_modifys)

    def calc_z_shapes(self, n_channel, input_resolution, n_block):
        z_shapes = []

        for i in range(n_block - 1):
            # except the last block, each block halves H and W and doubles C
            input_resolution //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_resolution, input_resolution))

        input_resolution //= 2
        z_shapes.append((n_channel * 4, input_resolution, input_resolution))  # the last block halves H and W and quadruples C

        return z_shapes


class NetD(nn.Module):
    def __init__(self,
                 resolution=128,
                 num_channels=3,
                 fmap_max=512,  # max channel number in D
                 # f=[1, 2, 1] by default
                 f=None  # (Huge overload, if you don't have enough resources, please pass it as `f = None`)
                 ):
        """
            Notice: we only support input pic with height == width.
        """
        super().__init__()

        self.fmap_base = fmap_max * 4 * 4
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution / 2 <= self.fmap_base
        self.nf = lambda stage: min(int(self.fmap_base / (2.0 ** stage)), fmap_max)
        self.fromRGB = nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1)

        self.blur2d = Blur2d(f)
        # if height or weight >= 128, use avgpooling2d without increasing channels. else: use ordinary conv2d.
        self.avg_down_sample = nn.AvgPool2d(2)
        self.use_AvgPool = (self.resolution_log2 > 6)
        self.n_avg_down_sample = 0  # number of AvgPool layers
        self.conv_down_samples = None
        if self.use_AvgPool:
            self.n_avg_down_sample = int (np.log2(resolution/64))
            self.conv_down_samples = nn.ModuleList([nn.Conv2d(self.nf(x - 1), self.nf(x - 1), kernel_size=2, stride=2) for x in range(6, 2, -1)])
        else:
            self.conv_down_sample_first = nn.Conv2d(self.nf(self.resolution_log2 - 1), self.nf(self.resolution_log2 - 1), kernel_size=2, stride=2)
            self.conv_down_samples = nn.ModuleList([nn.Conv2d(self.nf(x - 1), self.nf(x - 1), kernel_size=2, stride=2) for x in range(self.resolution_log2 - 1, 2, -1)])

        self.conv_first = nn.Conv2d(self.nf(self.resolution_log2 - 1), self.nf(self.resolution_log2 - 1), kernel_size=3, padding=(1, 1))
        self.convs = nn.ModuleList([nn.Conv2d(self.nf(x), self.nf(x - 1), kernel_size=3, padding=(1, 1)) for x in range(self.resolution_log2 - 1, 2, -1)])
        self.conv_last = nn.Conv2d(self.nf(2), self.nf(1), kernel_size=3, padding=(1, 1))
        # calculate point
        self.dense0 = nn.Linear(self.fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        assert int(np.log2(input.shape[2])) == self.resolution_log2, "Wrong input resolution!"
        x = F.leaky_relu(self.fromRGB(input), 0.2)
        x = F.leaky_relu(self.conv_first(x), 0.2, inplace=True)
        if self.use_AvgPool:
            # input height or weight >= 128
            x = F.leaky_relu(self.avg_down_sample(self.blur2d(x)), 0.2, inplace=True)
            for i, conv in enumerate(self.convs):
                if i < self.n_avg_down_sample - 1:
                    # use avgpooling2d downsampling
                    x = F.leaky_relu(conv(x), 0.2, inplace=True)
                    x = F.leaky_relu(self.avg_down_sample(self.blur2d(x)), 0.2, inplace=True)
                else:
                    # use ordinary conv2d downsampling
                    x = F.leaky_relu(conv(x), 0.2, inplace=True)
                    x = F.leaky_relu(self.conv_down_samples[i-self.n_avg_down_sample + 1](x), 0.2, inplace=True)
        else:
            # input height or weight < 128
            x = self.conv_down_sample_first(x)
            for conv, conv_down_sample in zip(self.convs, self.conv_down_samples):
                x = F.leaky_relu(conv(x), 0.2, inplace=True)
                x = F.leaky_relu(conv_down_sample(x), 0.2, inplace=True)

        x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
        return x


if __name__ == '__main__':
    x = torch.ones((1, 3, 128, 128), dtype=torch.float32)
    netG = NetG(128)
    latent = torch.randn((1, 512), dtype=torch.float32)
    output = netG(x, latent)
    print(output.shape)
    netD = NetD()
    score = netD(output)
    print(score.shape)
