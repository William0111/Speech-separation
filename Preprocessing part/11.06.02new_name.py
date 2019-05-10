

import os

# for file in os.listdir('.'):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
#     if file[-2: ] == 'py':
#         continue   #过滤掉改名的.py文件
#     name = file.replace(' ', '')   #去掉空格
#     new_name = name[20: 30] + name[-4:]   #选择名字中需要保留的部分
#     os.rename(file, new_name)

            # dirfile_name='./flac' # 代码文件和需要修改的文件在同一个目录下
            #
            # filename_list = os.listdir(dirfile_name)
            #
            # num = 1
            # for file in filename_list:
            #     #print(str(file))
            #     # if file[0:9] == '.DS_Store':
            #     #     print (file)
            #     #     continue
            #     new_name = str(num)+'.flac'
            #     os.rename(file, new_name)
            #     num = num+1



    # name = file.replace('1', '')   #去掉空格
    # os.rename(file, name)
    #
    # print('After rename:')
    # print(str(file))
#
# num = 0
# for file in os.listdir('raw_data/'):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
#     if file[0:9] != 'raw_data_':
#         print (file)
#         continue   #只更改raw_data_
#     name = file.replace(' ', '')   #去掉空格
#     new_name = name[0: 9] + str(num)   #选择名字中需要保留的部分
#     os.rename(file, new_name)
#     num = num + 1

file_dir = './flac'

for root,dirs,files in os.walk(file_dir):
    # 设置路径到每个子文件夹，子子文件夹......
    os.chdir(root)
    i = 1
    # 遍历每个子文件夹，子子文件夹......中的每个文件
    for filespath in files:
        # 将原本的文件的后缀名提取出来，先以‘.’进行分割，然后用old_file_name_split[-1]提取出后缀名
        old_file_name_split = filespath.split('.')
        # 将新名称修改为1.txt, 2.txt, ......
        new_name = str(i) + '.' + old_file_name_split[-1]
        # 替换名称（注意，原本的名称不能有1.txt等，不然会替换失败）
        os.rename(filespath, new_name)
        i += 1
