'''
1、去除全白的切片
2、将三轴面下颌骨图像中的黑色和白色均填充为灰色
'''
import cv2 
import os
import json
import numpy as np

# fenge_old_list = ['ax_fenge','sag_fenge','cor_fenge']
# fenge_new_list = ['ax_fenge2','sag_fenge2','cor_fenge2']
# quesun_old_list = ['ax_quesun','sag_quesun','cor_quesun']
# quesun_new_list = ['ax_quesun2','sag_quesun2','cor_quesun2']

old_path = 'C:\\Users\\Landis\\Desktop\\clinical_quesun1'
new_path = 'C:\\Users\\Landis\\Desktop\\clinical_quesun2'

'''只适用于ax的处理 因为sag和cor在quesun文件夹的中间部分也会出现空白切片'''
old_list = os.listdir(old_path)

for item in old_list:
    in_path = os.path.join(old_path, item)
    out_path = os.path.join(new_path, item)
    os.mkdir(out_path)

    filename = os.listdir(in_path)
    filename.sort(key=lambda x:int(x[:-4]))

    count = 0     # 设置变量用来作为保存后jpg图像的名称的
    # 开始遍历文件夹下的每张图像
    for i in filename:
        document = os.path.join(in_path, i)

        countname = str(count) # 将设置的变量数字，转换为字符串，作为名称使用
        countfullname = countname + '.jpg' # 后缀.jpg
        output_jpg_path = os.path.join(out_path, countfullname) # 设置保存每张图片的路径
        
        jpg_image = cv2.imread(document,cv2.CV_8UC1)
        jpg_image2 = cv2.imread(document,cv2.CV_8UC1)

        '''填充区域颜色'''
        jpg_height = int(jpg_image2.shape[0])
        jpg_width = int(jpg_image2.shape[1])

        for m in range(jpg_height):
            for n in range(jpg_width):
                if jpg_image2[m][n] == 255:
                    jpg_image2[m][n] = 223
                    jpg_image[m][n] = 223
                    continue
                elif jpg_image2[m][n] == 171:
                    jpg_image2[m][n] = 255
                    jpg_image[m][n] = 255
                    continue

        for m in range(jpg_height):
            for n in range(jpg_width):
                if jpg_image2[m][n] >= 250:
                    jpg_image2[m][n] = 0

        # 提取轮廓
        contours,heridency = cv2.findContours(jpg_image2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # 标记轮廓
        cv2.drawContours(jpg_image,contours,-1,(223,223,223),-1)

        # 转换为JPG格式
        cv2.imwrite(output_jpg_path, jpg_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        count += 1 # 为下一张图像的名称作准备，加1

        print("已完成{}文件夹下第{}张图像的处理".format(item, i))


'''适合三个轴面的统一方法: 统计fenge中的非空白范围, 应用在quesun中'''
# origin_path = 'C:\\Users\\yours\\Desktop'
# # 用于保存每个轴面非空白切片范围的字典,形如:{'134':[20,119],...}
# ax_dict = {}
# sag_dict = {}
# cor_dict = {}
# # 分别遍历三个分割文件夹，获取到每例CT非空白切片的范围保存在字典中
# for (index, value) in enumerate(fenge_old_list):
#     in_path = os.path.join(origin_path, value)
#     out_path = os.path.join(origin_path, fenge_new_list[index])
#     filename1 = os.listdir(in_path)

#     # 开始遍历文件夹下的每张图像
#     for i in filename1:
#         count = 0     # 设置变量用来作为保存后jpg图像的名称的
#         document = os.path.join(in_path, i)
#         filename2 = os.listdir(document)
#         filename2.sort(key=lambda x:int(x[:-4]))

#         outputpath = os.path.join(out_path, i) # 保存jpg图像的路径
#         os.mkdir(outputpath)

#         valid_list = []

#         for j in filename2:
#             image_document = os.path.join(document, j)
#             countname = str(count) # 将设置的变量数字，转换为字符串，作为名称使用
#             countfullname = countname + '.jpg' # 后缀.jpg
#             output_jpg_path = os.path.join(outputpath, countfullname) # 设置保存每张图片的路径
            
#             jpg_image = cv2.imread(image_document)
#             jpg_image2 = cv2.imread(image_document,cv2.CV_8UC1)

#             '''去除空白切片'''
#             max_pixel = np.max(jpg_image)
#             min_pixel = np.min(jpg_image)
#             if max_pixel == min_pixel:
#                 continue
#             else:
#                 valid_list.append(int(j[:-4]))
#                 '''填充区域颜色'''
#                 jpg_height = int(jpg_image2.shape[0])
#                 jpg_width = int(jpg_image2.shape[1])

#                 for m in range(jpg_height):
#                     for n in range(jpg_width):
#                         if jpg_image2[m][n] >= 250:
#                             jpg_image2[m][n] = 0

#                 # 提取轮廓
#                 contours,heridency = cv2.findContours(jpg_image2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#                 # 标记轮廓
#                 cv2.drawContours(jpg_image,contours,-1,(230,230,230),-1)

#                 # 转换为JPG格式
#                 cv2.imwrite(output_jpg_path, jpg_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#                 count += 1 # 为下一张图像的名称作准备，加1

#             print("已完成{}文件夹下第{}例第{}张图像的处理".format(value, i, j))

#         min_num = valid_list[0]
#         max_num = valid_list[-1]
#         if value == 'ax_fenge':
#             ax_dict[str(i)] = [min_num, max_num]
#         if value == 'sag_fenge':
#             sag_dict[str(i)] = [min_num, max_num]
#         if value == 'cor_fenge':
#             cor_dict[str(i)] = [min_num, max_num]

# with open('./dict.txt', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(ax_dict))
#     f.write(json.dumps(sag_dict))
#     f.write(json.dumps(cor_dict))

# # 分别遍历三个缺损文件夹，根据字典中保存的范围筛选切片
# for (index, value) in enumerate(quesun_old_list):
#     in_path = os.path.join(origin_path, value)
#     out_path = os.path.join(origin_path, quesun_new_list[index])
#     filename1 = os.listdir(in_path)

#     # 开始遍历文件夹下的每张图像
#     for i in filename1:
#         count = 0     # 设置变量用来作为保存后jpg图像的名称的
#         document = os.path.join(in_path, i)
#         filename2 = os.listdir(document)
#         filename2.sort(key=lambda x:int(x[:-4]))

#         outputpath = os.path.join(out_path, i) # 保存jpg图像的路径
#         os.mkdir(outputpath)

#         if value == 'ax_quesun':
#             min_num = ax_dict[str(i[0:3])][0]
#             max_num = ax_dict[str(i[0:3])][1]
#         if value == 'sag_quesun':
#             min_num = sag_dict[str(i[0:3])][0]
#             max_num = sag_dict[str(i[0:3])][1]
#         if value == 'cor_quesun':
#             min_num = cor_dict[str(i[0:3])][0]
#             max_num = cor_dict[str(i[0:3])][1]

#         for j in filename2:
#             image_document = os.path.join(document, j)
#             countname = str(count) # 将设置的变量数字，转换为字符串，作为名称使用
#             countfullname = countname + '.jpg' # 后缀.jpg
#             output_jpg_path = os.path.join(outputpath, countfullname) # 设置保存每张图片的路径
        

#             if int(j[:-4]) < min_num or int(j[:-4]) > max_num:
#                 continue
#             else:
#                 jpg_image = cv2.imread(image_document)
#                 jpg_image2 = cv2.imread(image_document,cv2.CV_8UC1)

#                 '''填充区域颜色'''
#                 jpg_height = int(jpg_image2.shape[0])
#                 jpg_width = int(jpg_image2.shape[1])

#                 for m in range(jpg_height):
#                     for n in range(jpg_width):
#                         if jpg_image2[m][n] >= 250:
#                             jpg_image2[m][n] = 0

#                 # 提取轮廓
#                 contours,heridency = cv2.findContours(jpg_image2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#                 # 标记轮廓
#                 cv2.drawContours(jpg_image,contours,-1,(230,230,230),-1)

#                 # 转换为JPG格式
#                 cv2.imwrite(output_jpg_path, jpg_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#                 count += 1 # 为下一张图像的名称作准备，加1

#             print("已完成{}文件夹下第{}例第{}张图像的处理".format(value, i, j))