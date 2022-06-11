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


class ImgPatches(nn.Module):
    def __init__(self, input_channels=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim, kernel_size=patch_size, stride=patch_size)
        
        def forward(self, img):
            patches = self.patch_embed(img).flatten(2).transpose(1,2)
            return patches

def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


class Encoder_Block():
	def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.1):
		super().__init__()
		self.ln1 = nn.LayerNorm(dim)
		self.attn = Attention(dim, heads, drop_rate, drop_rate)
		self.ln1 = nn.LayerNorm(dim)
		self.mlp = MLP(dim, dim*mlp_ratio, drop=drop_rate)

	def forward(self, x):
		x1 = self.ln1(x)
		x = x + self.attn(x1)
		x2 = self.ln2(x)
		x = x + self.mlp(x2)
		return x

class TransformerEncoder(nn.Module):
	def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
		super().__init__()
		self.Encoder_Blocks = nn.ModuleList([
		Encoder_Block(dim, heads, mlp_ratio, drop_rate)
		for i in range(depth)])

	def forward(self, x):
		for Encoder_Block in self.Encoder_Blocks:
			x = Encoder_Block(x)
		return x

class Generator(nn.Module):
    """ implement the GAN wiht the previous transformer block"""
    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384,
                 heads=4, mlp_ratio=4, drop_rate=0.):#,device=device):
        super(Generator, self).__init__()

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ration
        self.droprate_rate = drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size ** 2)* self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2,
                                                               384//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2,
                                                               384//16))

        self.TransformerEncoder_encoder1 =
        TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads,
                           mlp_ratio=self.mlp_ratio,
                           drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 =
        TransformerEncoder(depth=self.depth2, dim=self.dim//4,
                           heads=self.heads, mlp_ratio=self.mlp_ratio,
                           drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 =
        TransformerEncoder(depth=self.depth3, dim=self.dim//16,
                           heads=self.heads, mlp_ratio=self.mlp_ratio,
                           drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)
        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x,H,W = UpSampling(x,H,W) 
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x,H,W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        return x


class Discriminator(nn.Module):
    """
    Implementation of a TransGAN discriminator

    """
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channel=3,
                num_classes=1, dim=384, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0)
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size')
        num_patches = (image_size//patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patch+1,
                                                             dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads, mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x

class Generator_(nn.Module):
    """
    Implementation of a simple GAN Generator with linear layers.
    
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
            layers = [nn.Conv1d(in_feat, out_feat, kernel_size =1)]
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
    Implementation of a simple GAN Discriminator.
    
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
            modules.add_module('Conv1d_{}'.format(i), nn.linear(n_units, n_units))
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



class Residual(nn.Module):
    """the Residual blocks try to implement in the conv residual in the future"""
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size =1, stride = strides)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels, kernel_size=1, stride = strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2s(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
