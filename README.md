# Mandible_Reconstruction
## 1、目录介绍
* **checkpoints**：保存训练节点文件夹
* **data**：输入网络之前对数据预处理文件夹
* **datasets**：数据集文件夹
* **models**：网络模型文件夹
* **options**：模型训练和测试的选项文件夹
* **preprocess**：获取网络要求的数据格式文件夹
* **results**：保存测试结果文件夹
* **scripts**：脚本文件夹（不常用）
* **util**：可视化文件夹（不常用）
* **train.py**：训练入口
* **test.py**：测试入口
* **requirements.txt**：所需环境（服务器docker环境比较杂，所以都拷贝了）
## 2、测试步骤
### （1）准备测试数据集
* 准备一个文件夹Test，其中自定义一个数据集文件夹data（命名随意）和“preprocess”文件夹中的combine_A_and_B.py；
* 数据集文件夹data中又包含```A/test```（存放完整图片，如果是临床数据集没有完整图片那么也存放缺损图片）和```B/test```（存放缺损图片），这两个文件夹中的图片需要一一对应，具体实现代码可参考“preprocess”文件夹中的python文件；
* 返回Test目录，在地址栏中输入```cmd```进入命令行，输入```python combine_A_and_B.py --fold_A data/A --fold_B data/B --fold_AB data```会自动构造出网络要求的数据格式并保存在data文件夹下，再将其放入datasets文件夹下，已给出一个简单示例；
### （2）进行测试
* 目前checkpoints中给出的是效果较好的一次训练集模型，利用该文件进行测试即可；
* 通过命令行进入到Mandible_Reconstruction目录下，输入```python3 test.py --dataroot ./datasets/[数据集名称，例如:data] --name [保存的训练集模型，例如:mandible] --model pix2pix --preprocess scale_width_and_crop --direction BtoA --gpu_ids [训练使用的显卡编号，例如:0]```；
* 测试完成结果将会保存在results文件夹中；
## 3、训练步骤（如果需要的话）
### （1）准备训练数据集
* 准备一个文件夹Train，其中自定义一个数据集文件夹data（命名随意）和“preprocess”文件夹中的combine_A_and_B.py；
* 数据集文件夹data中又包含```A/train```（存放完整图片）和```B/train```（存放缺损图片），这两个文件夹中的图片需要一一对应，具体实现代码可参考“preprocess”文件夹中的python文件；
* 返回Train目录，在地址栏中输入```cmd```进入命令行，输入```python combine_A_and_B.py --fold_A data/A --fold_B data/B --fold_AB data```会自动构造出网络要求的数据格式并保存在data文件夹下，再将其放入datasets文件夹下，已给出一个简单示例；
### （2）修改代码
* 网络模型的相关代码大部分都在```models/networks.py```和```models/pix2pix_model.py```中；
* 训练参数的相关代码大部分都在```options/base_options.py```和```options/train_options.py```中；
### （3）进行训练
* 通过```python3 -m visdom.server --hostname=[服务器IP，例如:"192.168.7.188"] -port 8097```开启可视化工具，开启后在浏览器中输入192.168.7.188:8097就可以看到训练过程中损失函数的变化和生成的结果；
* 通过命令行进入到Mandible_Reconstruction目录下，输入```python3 train.py --dataroot ./datasets/[数据集名称，例如:data] --name [训练集模型的名称，例如:mandible] --model pix2pix --preprocess scale_width_and_crop --direction BtoA --gpu_ids [训练使用的显卡编号，例如:0]```；
* 训练完成模型结果将会保存在checkpoints文件夹中；
