import torch
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
from torchvision import transforms


class Pix2PixModel(BaseModel):
    """ 
    这个类实现了pix2pix模型, 用于学习从输入图像到给定的成对数据的输出图像的映射.

    该模型训练要求：
        一个训练模式为对齐的数据集: '--dataset_mode aligned' dataset.
        一个默认使用U-Net256的生成器: '--netG unet256',
        一个使用PatchGAN的判别器: '--netD basic',
        一个原始论文中使用的交叉熵目标函数: '--gan_mode' vanilla GAN loss.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        添加新的特定于数据集的选项，并重写现有选项的默认值.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        对于pix2pix不使用图像缓冲区
        训练的损失函数: GAN Loss + lambda_L1 * ||G(A)-B||_1
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失和评价指标。训练/测试脚本将调用 <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_Sep', 'D_real', 'D_fake']
        # 指定要保存/显示的图像。训练/测试脚本将调用 <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # 指定要保存到磁盘的模型。训练/测试脚本将调用 <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # 测试时只加载生成器
            self.model_names = ['G']
        # 定义生成器
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 定义局部判别器
        if self.isTrain:  
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) 
        
        if self.isTrain:
            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSep = networks.AdditionalEncoder().to(self.device)

            # 初始化优化器
            # 调度器将通过函数 <BaseModel.setup> 自动创建.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input):
        """
        从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身和它的元数据信息

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)       # 缺损图像
        self.real_B = input['B' if AtoB else 'A'].to(self.device)       # 完整图像
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """
        前向传播
        被两个函数调用：<optimize_parameters> and <test>.
        """
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """计算判别器GAN的loss"""
        # 局部判别器+全局判别器
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        local_pred_fake, global_pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(local_pred_fake, False) + self.criterionGAN(global_pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        local_pred_real, global_pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(local_pred_real, True) + self.criterionGAN(global_pred_real, True)

        # 联合损失+计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """计算生成器GAN loss、L1 loss和Perceptual loss"""
        # 1、cGANloss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        local_pred_fake, global_pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = (self.criterionGAN(local_pred_fake, True) + self.criterionGAN(global_pred_fake, True)) * 0.1
        '''此处可以把超参数调为0, 效果会更好一些'''
        # self.loss_G_GAN = (self.criterionGAN(local_pred_fake, True) + self.criterionGAN(global_pred_fake, True)) * 0

        # 2、生成图像和真实图像间L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # 3、生成图像和真实图像间潜在空间一致性Sep loss
        self.loss_G_Sep = self.criterionSep(self.fake_B, self.real_B) * 10

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Sep
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # 生成假图像: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # 设置为要求梯度，使得D可以反向传播
        self.optimizer_D.zero_grad()     # 设置判别器D的梯度为0
        self.backward_D()                # 计算判别器D的梯度
        self.optimizer_D.step()          # 更新判别器D的权重

        # update G
        self.set_requires_grad(self.netD, False)  # 当优化生成器G时不要求判别器D的梯度
        self.optimizer_G.zero_grad()        # 设置生成器G的梯度为0
        self.backward_G()                   # 计算生成器G的梯度
        self.optimizer_G.step()             # 更新生成器G的权重
