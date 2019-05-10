

import os

def rename():
    count=10000
    path="/home/william/Documents/code/After0303/train-clean-100/"
    filelist=os.listdir(path)#该文件夹下所有的文件（包括文件夹）
    for files in filelist:#遍历所有文件
        Olddir=os.path.join(path,files)#原来的文件路径
        if os.path.isdir(Olddir):#如果是文件夹则跳过
            continue
        filename=os.path.splitext(files)[0]#文件名
        filetype=os.path.splitext(files)[1]#文件扩展名
        Newdir=os.path.join(path,str(count)+filetype)#新的文件路径
        os.rename(Olddir,Newdir)#重命名
        count+=1

rename()