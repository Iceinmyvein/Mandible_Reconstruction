"""
这个包包含所有与数据加载和预处理相关的模块
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    在文件中, 名为DatasetNameDatset()的类将被实例化。它必须是BaseDatset的子类且不区分大小写。
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """返回数据集类的静态方法"""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """给定选项创建一个数据集

    这个函数包装了CustomDatasetDataLoader类, 是这个包和'train.py'/'test.py'之间的主接口

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """执行多线程数据加载的Dataset类的包装器类"""

    def __init__(self, opt):
        """类的初始化

        Step 1: 创建名称为[dataset_mode]的数据集实例
        Step 2: 创建多线程数据加载器
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        '''数据读取接口: torch.utils.data.DataLoader(数据集, batch_size, 是否在每个epoch重新打乱数据(默认为False), 用多少个子进程加载数据(默认为0))'''
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """返回数据集中的数据数量"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """返回一批数据"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
    
    def getitem(self, index):
        '''返回某索引值的数据'''
        return self.dataset.__getitem__(index)
