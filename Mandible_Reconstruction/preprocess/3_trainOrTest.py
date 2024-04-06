'''
将45例数据按照8:2的比例分为训练集train和测试集test
    A(完整数据 36例): train + test
    B(缺损数据 9例): train + test
'''

import os
import cv2
import numpy as np

image_train_index = ['134', '135', '136', '137', '138', '139', '140', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '157', '159', '160', '163', '164', '165', '166', '168', '169', '170', '173', '175', '177', '179', '181', '184', '185', '189']
image_test_index = ['156', '158', '162', '171', '176', '178', '186', '188', '190']

# 原始完整数据和缺损数据的路径
wanzheng_path = 'C:\\Users\\Landis\\Desktop\\ax_fenge2'
quesun_path = 'C:\\Users\\Landis\\Desktop\\ax_quesun2'
quesun_ewai = ['AND1-1','AND1-2','AND1-3','AND2-1','AND2-2','AND2-3','AND4']


# 输出路径
A_train_path = 'C:\\Users\\Landis\\Desktop\\xxx\\test\\A\\train'
A_test_path = 'C:\\Users\\Landis\\Desktop\\xxx\\test\\A\\test'
B_train_path = 'C:\\Users\\Landis\\Desktop\\xxx\\test\\B\\train'
B_test_path = 'C:\\Users\\Landis\\Desktop\\xxx\\test\\B\\test'


# 实现A/test和B/test的数据(每一例数据随机选取一个区域，全部处理)
# index = 0
# test_count = 1
# for i in image_test_index:
#     quesun_document = os.path.join(quesun_path, i)
#     wanzheng_document = os.path.join(wanzheng_path, i)

#     wanzheng_piclist = os.listdir(wanzheng_document)
#     wanzheng_piclist.sort(key=lambda x:int(x[:-4]))
#     wanzheng_length = len(wanzheng_piclist)

#     # 1-7分别对应1-1、1-2、1-3、2-1、2-2、2-3、4
#     random_array = [4,7,1,5,3,6,2,1]
#     j = random_array[index]
#     quesun_all_document = quesun_document + quesun_ewai[j-1]

#     quesun_piclist = os.listdir(quesun_all_document)
#     quesun_piclist.sort(key=lambda x:int(x[:-4]))
#     quesun_length = len(quesun_piclist)
    
#     for k in quesun_piclist:
#         quesun_sub_document = os.path.join(quesun_all_document, k)
#         quesun_image_array = cv2.imread(quesun_sub_document)

#         wanzheng_sub_document = os.path.join(wanzheng_document, k)
#         wanzheng_image_array = cv2.imread(wanzheng_sub_document)


#         save_number = str(test_count)
#         save_name = i + '-' + str(j) + '-' + save_number + '.jpg'
#         save_B_test_path = os.path.join(B_test_path, save_name)
#         save_A_test_path = os.path.join(A_test_path, save_name)

#         cv2.imwrite(save_B_test_path, quesun_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#         cv2.imwrite(save_A_test_path, wanzheng_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#         test_count += 1
    
#         print("已完成第{}例中{}文件夹的第{}张图像的处理".format(i, j, k))
    
#     index += 1


# 实现A/test和B/test的数据(每一例数据每一处缺损都处理)
test_count = 1
for i in image_test_index:
    quesun_document = os.path.join(quesun_path, i)
    wanzheng_document = os.path.join(wanzheng_path, i)

    wanzheng_piclist = os.listdir(wanzheng_document)
    wanzheng_piclist.sort(key=lambda x:int(x[:-4]))
    wanzheng_length = len(wanzheng_piclist)

    for j in quesun_ewai:
        quesun_all_document = quesun_document + j

        quesun_piclist = os.listdir(quesun_all_document)
        quesun_piclist.sort(key=lambda x:int(x[:-4]))
        quesun_length = len(quesun_piclist)
        
        for k in quesun_piclist:
            quesun_sub_document = os.path.join(quesun_all_document, k)
            quesun_image_array = cv2.imread(quesun_sub_document)

            wanzheng_sub_document = os.path.join(wanzheng_document, k)
            wanzheng_image_array = cv2.imread(wanzheng_sub_document)

            difference = cv2.subtract(quesun_image_array, wanzheng_image_array)
            result = not np.any(difference)

            if(result == True):
                continue
            else:
                save_number = str(test_count)
                save_name = i + '-' + str(j) + '-' + save_number + '.jpg'
                save_B_test_path = os.path.join(B_test_path, save_name)
                save_A_test_path = os.path.join(A_test_path, save_name)

            cv2.imwrite(save_B_test_path, quesun_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(save_A_test_path, wanzheng_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            test_count += 1
        
            print("已完成第{}例中{}文件夹的第{}张图像的处理".format(i, j, k))
    


    
# 实现A/train和B/train的数据
# train_count = 1
# for i in image_train_index:
#     quesun_document = os.path.join(quesun_path, i)
#     wanzheng_document = os.path.join(wanzheng_path, i)

#     wanzheng_piclist = os.listdir(wanzheng_document)
#     wanzheng_piclist.sort(key=lambda x:int(x[:-4]))
#     wanzheng_length = len(wanzheng_piclist)

#     for j in quesun_ewai:
#         quesun_all_document = quesun_document + j

#         quesun_piclist = os.listdir(quesun_all_document)
#         quesun_piclist.sort(key=lambda x:int(x[:-4]))
#         quesun_length = len(quesun_piclist)
        
#         for k in quesun_piclist:
#             quesun_sub_document = os.path.join(quesun_all_document, k)
#             quesun_image_array = cv2.imread(quesun_sub_document)

#             wanzheng_sub_document = os.path.join(wanzheng_document, k)
#             wanzheng_image_array = cv2.imread(wanzheng_sub_document)

#             difference = cv2.subtract(quesun_image_array, wanzheng_image_array)
#             result = not np.any(difference)

#             if(result == True):
#                 continue
#             else:
#                 save_number = str(train_count)
#                 save_name = i + '-' + save_number + '.jpg'
#                 save_B_train_path = os.path.join(B_train_path, save_name)
#                 save_A_train_path = os.path.join(A_train_path, save_name)

#                 cv2.imwrite(save_B_train_path, quesun_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#                 cv2.imwrite(save_A_train_path, wanzheng_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#                 train_count += 1
        
#             print("已完成第{}例中{}文件夹的第{}张图像的处理".format(i, j, k))
        