%----------------------------------------------------------------
%  Patch augmentation by cropping a certain number of image patches 
%  from each image using Maximal Poisson-disk Sampling (MPS)
%----------------------------------------------------------------

clc;
clear all;
kCropHeight = 240;  % the height of patch
kCropWidth = 240;  % the width of patch
kCropNum = 200;  % the number of cropped patches for each image
mps_data = load('mps/maxmps_200.pts');
rng('shuffle');  % used for randi

mainDir =  "C:\Users\jalouam\Desktop\ML\NIvsCG\datasets\"
subDirs = ["personal"; "PRCG"]

test_freq = 4;  %how often to use an image as a validation image. every 5th
valid_freq = 5; %6th image

for i = 1:2
    files = dir(strcat(mainDir, 'full\', subDirs{i})) %get all the image files in the directory 
    files = files(3:end) %ignore the '.' and '..' directories
    counter = 1; %counter for determining what type of image it is (test/valid etc...
    for k = 1:800 %go through all files in this folder
        type = 'train\';
        if counter == test_freq
            type = 'test\';
        end
        if counter == valid_freq
            type = 'valid\';
            counter = 0;
        end
        image_name = files(k).name;
        img = imread(char(strcat(mainDir,'full\', subDirs{i}, '\', image_name)));
        pos_flag = zeros(4,1);  % four corners
        for x = 1:kCropNum
            [cut pos_flag]= imageMpsCrop(img, pos_flag, mps_data, x, kCropWidth, kCropHeight); % mps sampling
            image_name_split = strsplit(image_name, '.');
            imwrite(cut, char(strcat(mainDir, 'patches\', type, subDirs{i}, '\', image_name_split{1}, '-', num2str(x), '.bmp')));
        end
        disp(sprintf("index: %d, img#: %d, counter: %d, type: %s", i,k,counter,type))
        counter  = counter +  1;
    end
    
end
