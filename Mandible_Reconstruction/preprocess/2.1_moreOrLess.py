'''
第二步filter中可能会出现quesun文件夹中前面多出切片或后面多出切片的现象:
    1、对于前面多出的切片, 用该脚本删去并重新命名
    2、对于后面多出的切片, 直接手动删除
'''
import os

'''删除quesun文件夹前面的切片并重命名'''
path = 'C:\\Users\\Landis\\Desktop\\clinical_quesun\\zhenxiuqiong'
# and_list = ['AND1-1','AND1-2','AND1-3','AND2-1','AND2-2','AND2-3','AND4']

# for value in and_list:
#     target_path = path + value
#     pic_name = os.listdir(target_path)
#     pic_name.sort(key=lambda x:int(x[:-4]))
    
#     # 删除前面多余的切片
#     for index in range(3):
#         pic_path = os.path.join(target_path, pic_name[index])
#         os.remove(pic_path)
    
#     # 对剩下的切片重命名
#     i = 0
#     new_name = os.listdir(target_path)
#     new_name.sort(key=lambda x:int(x[:-4]))
#     for item in new_name:
#         os.rename(os.path.join(target_path, item), os.path.join(target_path, (str(i)+'.jpg')))
#         i += 1
    
#     print('已完成{}文件夹的处理'.format(value))


pic_name = os.listdir(path)
pic_name.sort(key=lambda x:int(x[:-4]))

# 删除前面多余的切片
for index in range(20):
    pic_path = os.path.join(path, pic_name[index])
    os.remove(pic_path)

# 对剩下的切片重命名
i = 0
new_name = os.listdir(path)
new_name.sort(key=lambda x:int(x[:-4]))
for item in new_name:
    os.rename(os.path.join(path, item), os.path.join(path, (str(i)+'.jpg')))
    i += 1

print('已完成')

'''删除fenge文件夹前面的切片并重命名'''
# path = 'C:\\Users\\yours\\Desktop\\sag_fenge2\\157'

# pic_name = os.listdir(path)
# pic_name.sort(key=lambda x:int(x[:-4]))

# # 删除前面多余的切片
# for index in range(50):
#     pic_path = os.path.join(path, pic_name[index])
#     os.remove(pic_path)

# # 对剩下的切片重命名
# i = 0
# new_name = os.listdir(path)
# new_name.sort(key=lambda x:int(x[:-4]))
# for item in new_name:
#     os.rename(os.path.join(path, item), os.path.join(path, (str(i)+'.jpg')))
#     i += 1
    
