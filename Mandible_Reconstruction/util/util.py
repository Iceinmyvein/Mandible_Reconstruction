"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
import numpy


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_test_image(image_numpy, image_path, img_w, img_h):
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    img_w = int(img_w / 2)
    img_h = img_h

    # resize的处理方法 ===> 先resize到512*512,得到结果后,在resize到原始大小,但得到的结果依旧会被压缩一点点
    # if img_w != w or img_h != h:
    #     image_pil = image_pil.resize((img_w, img_h), Image.BICUBIC)

    # scale的处理方法 ===> 根据横纵比和对称裁剪到512*512,得到结果后,在对称地增加或删减空白区域得到原始大小
    img = cv2.cvtColor(numpy.asarray(image_pil), cv2.COLOR_RGB2GRAY)
    img = numpy.uint8(img)
    if w >= img_w:
        x_left = int(0 + (w - img_w) / 2)
        x_right = int(w - (w - img_w) / 2)
        img = img[x_left:x_right, :]
    else:
        x_left = int((img_w - w) / 2)
        x_right = int((img_w - w) / 2)
        img = cv2.copyMakeBorder(img,0,0,x_left,x_right,cv2.BORDER_REPLICATE)
    
    if h >= img_h:
        y_top = int(0 + (h - img_h) / 2)
        y_bottom = int(h - (h - img_h) / 2)
        img = img[:, y_top:y_bottom]
    else:
        y_top = int((img_h - h) / 2)
        y_bottom = int((img_h - h) / 2)
        img = cv2.copyMakeBorder(img,y_top,y_bottom,0,0,cv2.BORDER_REPLICATE)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    img.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
