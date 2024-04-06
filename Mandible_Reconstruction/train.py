"""
图像到图像翻译通用训练脚本

该脚本适用于：
    各种模型(带有'--model'选项: pix2pix, cyclegan, colorization)
    不同的数据集(带有选项'--dataset_mode': aligned, unaligned, single, colorization)
    你需要指定数据集('--dataroot')
    实验名称('--name')
    模型('--model')

它首先创建模型、数据集和给定选项的可视化工具。然后进行标准的网络训练。在训练过程中，它还可以可视化/保存图像，打印/保存丢失图，保存模型。

该脚本支持继续/恢复培训。使用“--continue_train”恢复之前的训练。

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    dataset = create_dataset(opt)  # 创建一个给定opt.dataset_mode和其他选项的数据集
    dataset_size = len(dataset)    # 获取数据集图像的数量
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # 创建一个给定opt.model和其他选项的模型
    model.setup(opt)               # 常规设置:加载和打印网络并创建调度器
    visualizer = Visualizer(opt)   # 创建一个显示/保存图像和图形的可视化工具
    total_iters = 0                # 训练迭代的总次数

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 不同epoch的外循环; 通过<epoch_count>, <epoch_count>+<save_latest_freq>来保存模型
        epoch_start_time = time.time()  # 每个epoch的计时器
        iter_data_time = time.time()    # 每次迭代加载数据的计时器
        epoch_iter = 0                  # 当前epoch的训练迭代次数，每个epoch重置为0
        visualizer.reset()              # 重置可视化工具：确保它至少每隔一次epoch将结果保存为HTML
        model.update_learning_rate()    # 在每个epoch开始时更新学习速率

        for i, data in enumerate(dataset):  # 一个epoch的内循环
            iter_start_time = time.time()  # 每次迭代计算的计时器
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
 
            model.set_input(data)         # 从数据集中解包数据并应用预处理
            model.optimize_parameters()   # 计算损失函数，得到梯度，更新网络权重

            if total_iters % opt.display_freq == 0:   # 在visdom上显示图像并将图像保存到HTML文件
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:       # 打印训练loss，绘制出图像，并将日志信息保存到磁盘
                losses = model.get_current_losses()     # 获取损失函数
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每<save_latest_freq>次迭代缓存最新的模型
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # 每<save_epoch_freq>个epoch缓存最新的模型
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
