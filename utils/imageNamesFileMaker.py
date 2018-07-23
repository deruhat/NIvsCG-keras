from os import listdir
from os.path import isfile, join
main_dir = '../datasets/patches/test/'
sub_dirs = ['PRCG/', 'personal/']
string = ""
for i,sub_dir in enumerate(sub_dirs):
	counter = 0;
	path = main_dir + sub_dir
	for file in listdir(path):
		if isfile(path + file):
			string = string + file + ' ' + str(i) + '\n'
		counter += 1
		if counter % 200 == 0:
			string = string + "200\n"

with open('../datasets/patches/test/filenames.txt', 'w') as f:
	f.write(string[0:-1])
