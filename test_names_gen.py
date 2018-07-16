import os

path1="personal"  
path2="prcg"

dirList1=os.listdir(path1)
dirList2=os.listdir(path2)
with open("output.txt", "w") as f:
    i = 1
    for filename in dirList1:
        if(i < 200):
            print ('personal/', filename, sep='')
            f.write('personal/' + filename + ' 0' + '\n') 
            i = i + 1
        else:
            print ('personal/', filename, sep='')
            f.write('personal/' + filename + ' 0' + '\n') 
            f.write('end' + '\n') 
            i = 1
    for filename in dirList2:
        if(i < 200):
            print ('prcg/', filename, sep='')
            f.write('prcg/' + filename + ' 1' + '\n') 
            i = i + 1
        else:
            print ('prcg/', filename, sep='')
            f.write('prcg/' + filename + ' 1' + '\n') 
            f.write('end' + '\n') 
            i = 1