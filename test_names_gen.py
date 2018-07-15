import os

path1="personal"  
path2="prcg"

dirList1=os.listdir(path1)
dirList2=os.listdir(path2)
with open("output.txt", "w") as f:
    for filename in dirList1:
        print ('personal/', filename, sep='')
        f.write('personal/' + filename + ' 0' + '\n') 
    for filename in dirList2:
        print ('prcg/', filename, sep='')
        f.write('prcg/' + filename + ' 1' + '\n') 