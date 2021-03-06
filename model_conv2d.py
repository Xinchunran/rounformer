import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat =  in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Generator_img(nn.Module):
    """
    Implementation of a Deep Convolutional GAN Generator (for sqare images ).

    """
    def __init__(self, img_size, channels, latent_dim):
        super(Generator_img, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
#            nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_img(nn.Module):
    """
    Implementation of a Deep Convolutional GAN Discriminator.

    """
    def __init__(self, channels, img_size):
        super(Discriminator_img, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128))

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias = False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim, out_shape, n_units=1024):
        super(Generator, self).__init__()
        self.out_shape = out_shape
        if (type(out_shape) is tuple and len(out_shape) > 1):
            out = int(np.prod(out_shape))
            self.tag = 1
        else:
            self.tag = 0
            out = int(out_shape)

        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=n_units, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=n_units), nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=512), nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256), nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=256, out_channels=out, kernel_size=4, stride=2, padding=1))

        self.output = nn.Tanh()

    def forward(self, X):
        X = self.model(X)
        return self.output(X)


class Discriminator(nn.Module):
    def __init__(self, inp_shape, n_units=1024):
        super(Discriminator, self).__init__()

        if (type(inp_shape) is tuple and len(inp_shape) > 1):
            inp = int(np.prod(inp_shape))
            self.tag = 1
        else:
            self.tag = 0
            inp = int(inp_shape)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=inp_shape, out_channels=256, kernel_size=4, stride =2, padding=1), nn.InstanceNorm1d(256, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(512, affine=True), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm1d(1024,affine=True), nn.LeakyReLU(0.2,
                                                               inplace=True)) 

        self.output = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, X):
        if self.tag:
            X = X.view(X.size(0), -1)
        validity = self.model(X)

        return validity

class Generator_(nn.Module):
    """
    Implementation of a simple GAN sicriminator.
    
    """   
    def __init__(self, latent_dim, out_shape, n_layers=4, n_units=512):
        super(Generator_, self).__init__()
        self.out_shape = out_shape
        if (type(out_shape) is tuple and len(out_shape) > 1):
            out = int(np.prod(out_shape))
            self.tag = 1
        else:
            self.tag = 0
            out = int(out_shape)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            #layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            #layers.append(nn.LeakyReLU())
            layers.append(nn.Sigmoid())
            return layers

        modules = nn.Sequential(*block(latent_dim, n_units, normalize=False))

        for i in range(n_layers):
            modules.add_module('linear_{}'.format(i), nn.Linear(n_units, n_units))
            #modules.add_module('Sigmoid_{}'.format(i), nn.Sigmoid())
            #modules.add_module('leakyRel_{}'.format(i), nn.LeakyReLU())
            modules.add_module('GELU_{}'.format(i), nn.GELU())
        modules.add_module('linear_{}'.format(n_layers+1), nn.Linear(n_units, out))
        self.model = modules

    def forward(self, z):
        out = self.model(z)
        #out =  z.permute(2,0 , 1).clone().contiguous()
        #out = self.model(out)
        if self.tag:
            out = out.view(out.size(0), *self.out_shape)
        return out


class Discriminator_(nn.Module):
    """
    Implementation of a simple GAN sicriminator.
    
    """
    def __init__(self, inp_shape, n_layers=4, n_units=512):
        super(Discriminator_, self).__init__()

        if (type(inp_shape) is tuple and len(inp_shape) > 1):
            inp = int(np.prod(inp_shape))
            self.tag = 1
        else:
            self.tag = 0
            inp = int(inp_shape)
        
        modules = nn.Sequential()
        modules.add_module('linear', nn.Linear(inp, n_units))
        modules.add_module('leakyRelu', nn.LeakyReLU())
        for i in range(n_layers):
            modules.add_module('Conv1d_{}'.format(i), nn.Linear(n_units, n_units))
            #modules.add_module('linear_{}'.format(i), nn.Linear(n_units, n_units))
            modules.add_module('tanh_{}'.format(i), nn.Tanh())
            #modules.add_module('Sigmoid_{}'.format(i), nn.Sigmoid())
            #modules.add_module('GELU_{}'.format(i), nn.GELU())
        modules.add_module('linear_{}'.format(n_layers+1), nn.Linear(n_units, 1))
        self.model = modules

    def forward(self, img):
        if self.tag:
            img = img.view(img.size(0), -1)
        validity = self.model(img)

        return validity



class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
