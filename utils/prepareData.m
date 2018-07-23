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
% image_name = 'set2-arch-27.bmp';

[P, F] = subdir('input-data');

PersonalNames = cat(1,F{1});
disp(PersonalNames(1));
PRCGNames = cat(1,F{2});
disp(PRCGNames(1));

% begin MPS
pos_flag = zeros(4,1);  % four corners

for k = 1:800 % 800 images in each dataset
    % Modify the dataset input and output to correspond with desired data
    img = imread(char(strcat('input-data/prcg/',string(PRCGNames(k)))));
    image_name = char(PRCGNames(k));
    for x = 1:kCropNum
        [cut, pos_flag]= imageMpsCrop(img, pos_flag, mps_data, x, kCropWidth, kCropHeight); % mps sampling
        image_name_split = strsplit(image_name, '.');
        imwrite(cut, strcat('output-data/prcg/',image_name_split{1}, '-', num2str(x), '.bmp'));
    end
end


