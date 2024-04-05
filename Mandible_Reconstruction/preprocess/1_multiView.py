'''
通过DICOM图像转换得到完整数据集和缺损数据集的横断面、矢状面和冠状面的JPG图像
'''
import pydicom
import numpy as np
import os
import cv2

# 分割Dicom图像文件夹
# fenge_file = 'D:\\data\\data\\fenge'
# 缺损Dicom图像文件夹
quesun_file = 'C:\\Users\\Landis\\Desktop\\clinical_data\\original_segment_excise'
# 各轴面存放JPG图像的文件夹
# ax_fenge_file = 'C:\\Users\\Landis\\Desktop\\ax_fenge'
ax_quesun_file = 'C:\\Users\\Landis\\Desktop\\clinical_quesun'
# sag_fenge_file = 'C:\\Users\\Landis\\Desktop\\sag_fenge'
# sag_quesun_file = 'C:\\Users\\Landis\\Desktop\\sag_quesun'
# cor_fenge_file = 'C:\\Users\\Landis\\Desktop\\cor_fenge'
# cor_quesun_file = 'C:\\Users\\Landis\\Desktop\\cor_quesun'

'''分割图像处理'''
# fenge_case = os.listdir(fenge_file)
# for fcase in fenge_case:
#     # 创建子文件夹
#     ax_case_file = os.path.join(ax_fenge_file, fcase)
#     os.mkdir(ax_case_file)
#     sag_case_file = os.path.join(sag_fenge_file, fcase)
#     os.mkdir(sag_case_file)
#     cor_case_file = os.path.join(cor_fenge_file, fcase)
#     os.mkdir(cor_case_file)

#     # 加载Dicom文件
#     files = []
#     target_file = os.path.join(fenge_file, fcase)
#     filename = os.listdir(target_file)
#     for fname in filename:
#         fname = os.path.join(target_file, fname)
#         files.append(pydicom.dcmread(fname))

#     # 像素方面，假设所有切片都相同
#     ps = files[0].PixelSpacing
#     ss = files[0].SliceThickness
#     ax_aspect = ps[1] / ps[0]
#     sag_aspect = ps[1] / ss
#     cor_aspect = ss / ps[0]

#     # 创建三维数组
#     img_shape = list(files[0].pixel_array.shape)
#     img_shape.append(len(files))
#     img3d = np.zeros(img_shape)

#     # 用files中的图像填充3D数组
#     for i, s in enumerate(files):
#         img2d = s.pixel_array
#         img3d[:, :, i] = img2d
    

#     for num in range(img_shape[2]):
#         # 保存横断面结果
#         ax = img3d[:, :, num]
#         ax_rescaled_image = (np.maximum(ax,0)/ax.max())*255 # 将像素值规范到0~255
#         ax_final_image = np.uint8(ax_rescaled_image)
#         name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
#         fullname = name + '.jpg'    # 后缀.jpg
#         ax_path = os.path.join(ax_case_file, fullname)
#         cv2.imwrite(ax_path, ax_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#     for num in range(img_shape[1]):
#         # 保存矢状面结果
#         sag = img3d[:, num, :]
#         # 将矩阵逆时针旋转90°
#         sag = np.rot90(sag)
#         sag_rescaled_image = (np.maximum(sag,0)/sag.max())*255 # 将像素值规范到0~255
#         sag_final_image = np.uint8(sag_rescaled_image)
#         sag_final_image = cv2.resize(sag_final_image, (0, 0), fx=1, fy=(1/sag_aspect))
#         name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
#         fullname = name + '.jpg'    # 后缀.jpg
#         sag_path = os.path.join(sag_case_file, fullname)
#         cv2.imwrite(sag_path, sag_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#     for num in range(img_shape[0]):
#         # 保存冠状面结果
#         cor = img3d[num, :, :]
#         # 将矩阵逆时针旋转90°
#         cor = np.rot90(cor)
#         cor_rescaled_image = (np.maximum(cor,0)/cor.max())*255 # 将像素值规范到0~255
#         cor_final_image = np.uint8(cor_rescaled_image)
#         cor_final_image = cv2.resize(cor_final_image, (0, 0), fx=1, fy=cor_aspect)
#         name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
#         fullname = name + '.jpg'    # 后缀.jpg
#         cor_path = os.path.join(cor_case_file, fullname)
#         cv2.imwrite(cor_path, cor_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#     print('【分割】已完成{}文件夹的处理'.format(fcase))


'''缺损图像处理'''
quesun_case = os.listdir(quesun_file)
for qcase in quesun_case:
    # 创建子文件夹
    ax_case_file = os.path.join(ax_quesun_file, qcase)
    os.mkdir(ax_case_file)
    # sag_case_file = os.path.join(sag_quesun_file, qcase)
    # os.mkdir(sag_case_file)
    # cor_case_file = os.path.join(cor_quesun_file, qcase)
    # os.mkdir(cor_case_file)

    # 加载Dicom文件
    files = []
    target_file = os.path.join(quesun_file, qcase)
    filename = os.listdir(target_file)
    for fname in filename:
        fname = os.path.join(target_file, fname)
        files.append(pydicom.dcmread(fname))

    # 像素方面，假设所有切片都相同
    ps = files[0].PixelSpacing
    ss = files[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # 创建三维数组
    img_shape = list(files[0].pixel_array.shape)
    img_shape.append(len(files))
    img3d = np.zeros(img_shape)

    # 用files中的图像填充3D数组
    for i, s in enumerate(files):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    for num in range(img_shape[2]):
        # 保存横断面结果
        ax = img3d[:, :, num]
        ax_rescaled_image = (np.maximum(ax,0)/ax.max())*255 # 将像素值规范到0~255
        ax_final_image = np.uint8(ax_rescaled_image)
        name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
        fullname = name + '.jpg'    # 后缀.jpg
        ax_path = os.path.join(ax_case_file, fullname)
        cv2.imwrite(ax_path, ax_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # for num in range(img_shape[1]):
    #     # 保存矢状面结果
    #     sag = img3d[:, num, :]
    #     # 将矩阵逆时针旋转90°
    #     sag = np.rot90(sag)
    #     sag_rescaled_image = (np.maximum(sag,0)/sag.max())*255 # 将像素值规范到0~255
    #     sag_final_image = np.uint8(sag_rescaled_image)
    #     sag_final_image = cv2.resize(sag_final_image, (0, 0), fx=1, fy=(1/sag_aspect))
    #     name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
    #     fullname = name + '.jpg'    # 后缀.jpg
    #     sag_path = os.path.join(sag_case_file, fullname)
    #     cv2.imwrite(sag_path, sag_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # for num in range(img_shape[0]):
    #     # 保存冠状面结果
    #     cor = img3d[num, :, :]
    #     # 将矩阵逆时针旋转90°
    #     cor = np.rot90(cor)
    #     cor_rescaled_image = (np.maximum(cor,0)/cor.max())*255 # 将像素值规范到0~255
    #     cor_final_image = np.uint8(cor_rescaled_image)
    #     cor_final_image = cv2.resize(cor_final_image, (0, 0), fx=1, fy=cor_aspect)
    #     name = str(num)             # 将设置的变量数字，转换为字符串，作为名称使用
    #     fullname = name + '.jpg'    # 后缀.jpg
    #     cor_path = os.path.join(cor_case_file, fullname)
    #     cv2.imwrite(cor_path, cor_final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print('【缺损】已完成{}文件夹的处理'.format(qcase))
