import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.modules.utils import _pair
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.models as models
import random


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    返回一个正则化层

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, 我们使用可学习的仿射参数和跟踪运行统计信息(mean/stddev)
    For InstanceNorm, 我们不使用可学习的仿射参数并且不跟踪运行统计数据
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """
    返回一个学习率调度器

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- 存储所有的实验标志;需要是BaseOptions的子类
                              opt.lr_policy 学习率策略的名称: linear | step | plateau | cosine

    For 'linear', 在<opt.n_epochs>个epochs内保持相同的学习率, 在接下来<opt.n_epochs_decay>内线性衰减直到0
    对于其他调度器(step、plateau、cosine), 我们使用默认的PyTorch调度器
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 比例因子： normal, xavier, orthogonal

    在最初的pix2pix和CycleGAN论文中使用了“normal”, 但xavier和kaiming可能对于某些应用程序效果更好, 你可以自己试试
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    初始化网络：
        1、注册CPU/GPU设备
        2、初始化网络权重

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- 输入图像的通道数量
        output_nc (int) -- 输出图像的通道数量
        ngf (int) -- 最后一层卷积层的过滤器数量
        netG (str) -- 生成器结构名称: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- 网络中使用的正则化层的名称: batch | instance | none
        use_dropout (bool) -- 是否使用dropout层
        init_type (str)    -- 初始化方法的名称.
        init_gain (float)  -- 比例因子.
        gpu_ids (int list) -- GPU_id

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # SRA-GAN
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=4, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- 第一层卷积层的过滤器数量
        netD (str)         -- 判别器结构名称: basic | n_layers | pixel
        n_layers_D (int)   -- 判别器中卷积层的数量(当netD=='n_layers'时有效)
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    我们当前的实现提供了三种类型的鉴别器:
        [basic]: 在最初的pix2pix论文中描述的'PatchGAN'分类器。它可以分类70*70的小区域图像是真还是假。这样的补丁级鉴别器体系结构相比于一个完整的图像鉴别器具有较少的参数。

        [n_layers]: 使用此模式可以在鉴别器中指定conv层的数量(default=3 as used in [basic] (PatchGAN).)

        [pixel]: 可以对1*1的像素的真假进行分类

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """
    定义不同的GAN目标

    GANLoss类抽象了创建目标标签张量的需要, 它和输入的大小一样
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """
        创建与输入大小相同的标签张量

        Parameters:
            prediction (tensor) - - 通常是判别器的预测
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            一个附带了ground truth标签的标签张量, 以及输入的大小
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        给定判别器的输出和真值标签后计算损失.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """计算梯度惩罚损失, 用于WGAN-GP论文 https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# 重写Resnet生成器（Unet结构+Resnet101部件）
class ResnetGenerator(nn.Module):
    """
    基于Resnet的生成器, 由几个下采样/上采样操作之间的Resnet块组成

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        '''复现论文'''
        self.stem = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(input_nc, 64, kernel_size=7, padding=0, bias=use_bias),norm_layer(64),nn.LeakyReLU(True))

        self.downsample1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128),nn.LeakyReLU(True))
        self.downsample2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256),nn.LeakyReLU(True))

        self.resblock1 = nn.Sequential(
            ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        )
        self.convsample1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512),nn.LeakyReLU(True))
        self.resblock2 = nn.Sequential(
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        )
        self.convsample2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512),nn.LeakyReLU(True))
        self.resblock3 = nn.Sequential(
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        )
        self.convsample3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512),nn.LeakyReLU(True))
        self.resblock4 = nn.Sequential(
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        )
        self.convsample4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias))

        # 更改上采样方式
        self.convupsample4 = nn.Sequential(nn.ReLU(True),nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))

        self.y_pre_upblock4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.x_pre_upblock4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.upblock4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(True)
        )

        self.convupsample3 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.sw_convupsample3 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))

        self.y_pre_upblock3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.x_pre_upblock3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.upblock3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(True)
        )

        self.convupsample2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.sw_convupsample2 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))

        self.y_pre_upblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.x_pre_upblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )
        self.upblock2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(1024),
            nn.ReLU(True)
        )

        self.convupsample1 = nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        self.sw_convupsample1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))

        self.y_pre_upblock1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )
        self.x_pre_upblock1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )
        self.upblock1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128),nn.ReLU(True))
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(64),nn.ReLU(True))

        self.upstem = nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(64, output_nc, kernel_size=3, padding=0),nn.Tanh())

    def forward(self, x):
        # 编码部分
        x = self.stem(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x1 = self.resblock1(x)
        x2 = self.convsample1(x1)
        x2 = self.resblock2(x2)
        x3 = self.convsample2(x2)
        x3 = self.resblock3(x3)
        x4 = self.convsample3(x3)
        x4 = self.resblock4(x4)
        x5 = self.convsample4(x4)

        # 解码部分
        y5 = self.convupsample4(x5)
        # SW机制
        if random.random() < 0.3:
            y4 = self.upblock4(torch.cat([self.y_pre_upblock4(y5), self.x_pre_upblock4(x4)], 1))
            y4 = self.convupsample3(y4)
        else:
            y4 = self.sw_convupsample3(y5)
        if random.random() < 0.6:
            y3 = self.upblock3(torch.cat([self.y_pre_upblock3(y4), self.x_pre_upblock3(x3)], 1))
            y3 = self.convupsample2(y3)
        else:
            y3 = self.sw_convupsample2(y4)
        if random.random() < 0.9:
            y2 = self.upblock2(torch.cat([self.y_pre_upblock2(y3), self.x_pre_upblock2(x2)], 1))
            y2 = self.convupsample1(y2)
        else:
            y2 = self.sw_convupsample1(y3)
        y1 = self.upblock1(torch.cat([self.y_pre_upblock1(y2), self.x_pre_upblock1(x1)], 1))
        y = self.upsample2(y1)
        y = self.upsample1(y)
        y = self.upstem(y)
        return y

# 重写为Resnet101 Block
class ResnetBlock(nn.Module):
    """定义一个Resnet块"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        p = 0
        conv_block += [nn.Conv2d(dim, dim, kernel_size=1, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.Conv2d(dim, dim, kernel_size=1, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

# 以简单的方式重写生成器
class UnetGenerator(nn.Module):
    """创建一个基于U-Net的生成器"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 构造网络结构
        down_acfunc = nn.LeakyReLU(0.2, True)
        up_acfunc = nn.ReLU(True)

        self.downconv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),down_acfunc,norm_layer(64))
        self.resblock1 = nn.Sequential(
            PreActBottleneck(cin=3, cout=64, cmid=16, stride=2),
            PreActBottleneck(cin=64, cout=64, cmid=16),
            PreActBottleneck(cin=64, cout=64, cmid=16),
        )
        self.mat1 = MAT(img_size=256, in_channels=64, hidden_size=64, sr_ratio=16, num_layers=1, num_heads=1)
        # self.cbam1 = Origin_CBAM(channel=64)
        # self.downconv2 = nn.Sequential(down_acfunc, nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128))
        # self.resblock2 = nn.Sequential(
        #     PreActBottleneck(cin=64, cout=128, cmid=32, stride=2),
        #     PreActBottleneck(cin=128, cout=128, cmid=32),
        #     PreActBottleneck(cin=128, cout=128, cmid=32),
        # )
        self.resblock2 = nn.Sequential(
            PreActBottleneck(cin=64, cout=128, cmid=32, stride=2),
            PreActBottleneck(cin=128, cout=128, cmid=32),
            PreActBottleneck(cin=128, cout=128, cmid=32),
            PreActBottleneck(cin=128, cout=128, cmid=32),
        )
        self.mat2 = MAT(img_size=128, in_channels=128, hidden_size=128, sr_ratio=8, num_layers=1, num_heads=2)
        # self.cbam2 = Origin_CBAM(channel=128)
        # self.downconv3 = nn.Sequential(down_acfunc, nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        # self.resblock3 = nn.Sequential(
        #     PreActBottleneck(cin=128, cout=256, cmid=64, stride=2),
        #     PreActBottleneck(cin=256, cout=256, cmid=64),
        #     PreActBottleneck(cin=256, cout=256, cmid=64),
        #     PreActBottleneck(cin=256, cout=256, cmid=64),
        # )
        self.resblock3 = nn.Sequential(
            PreActBottleneck(cin=128, cout=256, cmid=64, stride=2),
            PreActBottleneck(cin=256, cout=256, cmid=64),
            PreActBottleneck(cin=256, cout=256, cmid=64),
            PreActBottleneck(cin=256, cout=256, cmid=64),
            PreActBottleneck(cin=256, cout=256, cmid=64),
            PreActBottleneck(cin=256, cout=256, cmid=64),
        )
        self.mat3 = MAT(img_size=64, in_channels=256, hidden_size=256, sr_ratio=4, num_layers=1, num_heads=4)
        # self.cbam3 = Origin_CBAM(channel=256)
        # self.downconv4 = nn.Sequential(down_acfunc, nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        # self.resblock4 = nn.Sequential(
        #     PreActBottleneck(cin=256, cout=512, cmid=128, stride=2),
        #     PreActBottleneck(cin=512, cout=512, cmid=128),
        #     PreActBottleneck(cin=512, cout=512, cmid=128),
        #     PreActBottleneck(cin=512, cout=512, cmid=128),
        #     PreActBottleneck(cin=512, cout=512, cmid=128),
        #     PreActBottleneck(cin=512, cout=512, cmid=128),
        # )
        self.resblock4 = nn.Sequential(
            PreActBottleneck(cin=256, cout=512, cmid=128, stride=2),
            PreActBottleneck(cin=512, cout=512, cmid=128),
            PreActBottleneck(cin=512, cout=512, cmid=128),
        )
        self.mat4 = MAT(img_size=32, in_channels=512, hidden_size=512, sr_ratio=2, num_layers=1, num_heads=8)
        # self.cbam4 = Origin_CBAM(channel=512)

        self.downconv5 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv6 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv7 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias))

        self.upconv7 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv6 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv5 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv4 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        self.upconv3 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128))
        self.upconv2 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(64))
        self.upconv1 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),nn.Tanh())

        self.dropout = nn.Dropout(0.5)

    
    # reshape: (B, n_patch, hidden_size) --> (B, hidden_size, h, w)
    def reshape_x(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0,2,1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

    def forward(self, x):
        x1 = self.downconv1(x)

        '''vit-cbam版本'''
        x1 = self.resblock1(x)
        mat_x1 = self.reshape_x(self.mat1(x1))
        x2 = self.resblock2(mat_x1)
        mat_x2 = self.reshape_x(self.mat2(x2))
        x3 = self.resblock3(mat_x2)
        mat_x3 = self.reshape_x(self.mat3(x3))
        x4 = self.resblock4(mat_x3)
        mat_x4 = self.reshape_x(self.mat4(x4))
        x5 = self.downconv5(mat_x4)
        x6 = self.downconv6(x5)
        x7 = self.downconv7(x6)

        y7 = self.upconv7(x7)
        y6 = self.upconv6(torch.cat([y7, x6], 1))
        y6 = self.dropout(y6)
        y5 = self.upconv5(torch.cat([y6, x5], 1))
        y5 = self.dropout(y5)
        y4 = self.upconv4(torch.cat([y5, mat_x4], 1))
        y3 = self.upconv3(torch.cat([y4, mat_x3], 1))
        y2 = self.upconv2(torch.cat([y3, mat_x2], 1))
        y1 = self.upconv1(torch.cat([y2, mat_x1], 1))

        return y1

# pix2pixGAN的生成器
class PixGenerator(nn.Module):
    """创建一个基于U-Net的生成器"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PixGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 构造网络结构
        down_acfunc = nn.LeakyReLU(0.2, True)
        up_acfunc = nn.ReLU(True)

        self.downconv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),down_acfunc,norm_layer(64))
        self.downconv2 = nn.Sequential(down_acfunc, nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128))
        self.downconv3 = nn.Sequential(down_acfunc, nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        self.downconv4 = nn.Sequential(down_acfunc, nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv5 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv6 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv7 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias))

        self.upconv7 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv6 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv5 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.upconv4 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        self.upconv3 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128))
        self.upconv2 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(64))
        self.upconv1 = nn.Sequential(up_acfunc,nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),nn.Tanh())

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x1 = self.downconv1(x)
        x2 = self.downconv2(x1)
        x3 = self.downconv3(x2)
        x4 = self.downconv4(x3)
        x5 = self.downconv5(x4.clone())
        x6 = self.downconv6(x5)
        x7 = self.downconv7(x6)

        y7 = self.upconv7(x7)
        y6 = self.upconv6(torch.cat([y7, x6], 1))
        y6 = self.dropout(y6)
        y5 = self.upconv5(torch.cat([y6, x5], 1))
        y5 = self.dropout(y5)
        y4 = self.upconv4(torch.cat([y5, x4], 1))
        y3 = self.upconv3(torch.cat([y4, x3], 1))
        y2 = self.upconv2(torch.cat([y3, x2], 1))
        y1 = self.upconv1(torch.cat([y2, x1], 1))

        return y1


class UnetSkipConnectionBlock(nn.Module):
    """定义带有跳级连接的U-Net子模块.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    # 默认不加Self-Attention层
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, add_self_attention=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        # 可变性卷积层
        deformconv = DeformConv2d(inc=input_nc, outc=inner_nc, bias=use_bias)
        # 平均池化层
        avgpool = nn.AvgPool2d(kernel_size=2, padding=0)
        # 最大池化层
        maxpool = nn.MaxPool2d(kernel_size=2, padding=0)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        # 自注意层
        # downselfattention = SelfAttention(inner_nc, 'rule')
        # upselfattention = SelfAttention(outer_nc*4, 'rule')

        # CBAM
        downcbam = CBAM(channel=inner_nc)
        upcbam = CBAM(channel=outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # down = [downconv]
            down = [deformconv, maxpool]
            up = [uprelu, upconv, nn.Tanh()]
            if add_self_attention:
                # model = down + [downselfattention] + [submodule] + up
                model = down + [downcbam] + [submodule] + up
            else:
                model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, deformconv, maxpool]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, deformconv, maxpool, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                if add_self_attention:
                    # model = down + [downselfattention] + [submodule] + [upselfattention] + up + [nn.Dropout(0.5)]
                    # 编码器解码器都加注意力
                    # model = down + [downcbam] + [submodule] + up  + [upcbam] + [nn.Dropout(0.5)]
                    # 只在解码器加注意力
                    # model = down + [submodule] + up + [upcbam] + [nn.Dropout(0.5)]
                    # 只在编码器加注意力
                    model = down + [downcbam] + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                if add_self_attention:
                    # model = down + [downselfattention] + [submodule] + [upselfattention] + up
                    # model = down + [downcbam] + [submodule] + up + [upcbam] 
                    # model = down + [submodule] + up + [upcbam]
                    model = down + [downcbam] + [submodule] + up 
                else:
                    model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


'''将patch判别器和1*1判别器写在一个类里面'''
class NLayerDiscriminator(nn.Module):
    """定义一个PatchGAN判别器"""

    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        nf_mult = 1
        nf_mult_prev = 1
        sequence = [nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

        net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*net)

    def forward(self, input):
        """Standard forward."""
        input_copy = input.clone()
        return self.model(input), self.net(input_copy)


class PixelDiscriminator(nn.Module):
    """定义一个1x1 PatchGAN判别器"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# 定义一个额外的编码器（用于高级特征误差估计）
class AdditionalEncoder(nn.Module):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d):
        super(AdditionalEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        down_acfunc = nn.LeakyReLU(0.2, True)
        
        self.downconv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=use_bias))
        self.cbam1 = Origin_CBAM(channel=64)
        self.downconv2 = nn.Sequential(down_acfunc, nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(128))
        self.cbam2 = Origin_CBAM(channel=128)
        self.downconv3 = nn.Sequential(down_acfunc,nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(256))
        self.cbam3 = Origin_CBAM(channel=256)
        self.downconv4 = nn.Sequential(down_acfunc,nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.cbam4 = Origin_CBAM(channel=512)
        self.downconv5 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv6 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(512))
        self.downconv7 = nn.Sequential(down_acfunc,nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias))

        self.mse_loss = nn.MSELoss()
    

    def forward(self, generated, target):
        g_x1 = self.downconv1(generated)

        cbam_g_x1 = self.cbam1(g_x1)
        g_x2 = self.downconv2(cbam_g_x1)
        cbam_g_x2 = self.cbam2(g_x2)
        g_x3 = self.downconv3(cbam_g_x2)
        cbam_g_x3 = self.cbam3(g_x3)
        g_x4 = self.downconv4(cbam_g_x3)
        cbam_g_x4 = self.cbam4(g_x4)
        g_x5 = self.downconv5(cbam_g_x4)
        g_x6 = self.downconv6(g_x5)
        g_x7 = self.downconv7(g_x6)



        t_x1 = self.downconv1(target)
        cbam_t_x1 = self.cbam1(t_x1)
        t_x2 = self.downconv2(cbam_t_x1)
        cbam_t_x2 = self.cbam2(t_x2)
        t_x3 = self.downconv3(cbam_t_x2)
        cbam_t_x3 = self.cbam3(t_x3)
        t_x4 = self.downconv4(cbam_t_x3)
        cbam_t_x4 = self.cbam4(t_x4)
        t_x5 = self.downconv5(cbam_t_x4)
        t_x6 = self.downconv6(t_x5)
        t_x7 = self.downconv7(t_x6)

        loss = self.mse_loss(g_x7, t_x7)
        return loss

# CBAM模块的实现
# 通道注意力
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

# 空间注意力
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# 顺序的CBAM（通道在前，空间在后）
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out = out.flatten(2).permute(0, 2, 1)
        return out

class Origin_CBAM(nn.Module):
    def __init__(self, channel):
        super(Origin_CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# Vision Transformer Block
# Embedding模块：patch_size为1，单纯改变通道数和形状
class Embeddings(nn.Module):
    def __init__(self, img_size, in_channels, hidden_size, patch_size=1, dropout_rate=0.1):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # patch编码
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.num_patches = n_patches
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embeddings(x)    # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)         # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, (H, W)

# MLP模块：参考CMT中的IR-FFT
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x

# Attention模块：轻量级自注意力，通过对K、V进行下采样实现
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=False, qk_scale=None, qk_ratio=1, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = hidden_size // qk_ratio

        self.query = nn.Linear(hidden_size, self.qk_dim, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.qk_dim, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio, groups=hidden_size, bias=True),
                nn.BatchNorm2d(hidden_size, eps=1e-5),
            )

    # def forward(self, x, H, W, relative_pos):
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            k = self.key(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.value(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.key(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.value(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        # x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

# Block模块：前面增加了局部表达单元，中间融入了CBAM模块和轻量级SA模块，后面是MLP单元
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super(Block, self).__init__()
        '''vit-cbam'''
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.cbam = CBAM(channel=dim)
        self.bn = nn.BatchNorm2d(dim, eps=1e-5)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        '''vit'''
        # self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # self.bn = nn.BatchNorm2d(dim, eps=1e-5)
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    # def forward(self, x, H, W, relative_pos):
    def forward(self, x, H, W):
        '''vit-cbam'''
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.cbam(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)

        '''vit'''
        # B, N, C = x.shape
        # cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        # x = self.proj(cnn_feat) + cnn_feat
        # x = x.flatten(2).permute(0, 2, 1)
        # x = x + self.attn(self.norm1(x), H, W)
        # x = x + self.mlp(self.norm2(x), H, W)
        return x

# Multil-Attention Transformer模块
class MAT(nn.Module):
    def __init__(self, img_size, in_channels, hidden_size, sr_ratio, num_layers, num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, patch_size=1,
                 dropout_rate=0.1, drop=0.0, attn_drop=0.0, qk_ratio=1):
        super(MAT, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, in_channels=in_channels, hidden_size=hidden_size, patch_size=patch_size, dropout_rate=dropout_rate)
        self.layer = nn.ModuleList()
        for _ in range(num_layers):
            layer = Block(dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                         attn_drop=attn_drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
            self.layer.append(copy.deepcopy(layer))
        # self.relative_pos = nn.Parameter(torch.randn(num_heads, self.embeddings.num_patches, self.embeddings.num_patches//sr_ratio//sr_ratio))

    def forward(self, x):
        embedding_output, (H, W) = self.embeddings(x)
        for layer_block in self.layer:
            # encoded = layer_block(embedding_output, H, W, self.relative_pos)
            encoded = layer_block(embedding_output, H, W)
        return encoded

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1,2,3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(16, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y
