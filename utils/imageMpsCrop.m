%-------------------------------------------------------------------------------------
% Cropping one image patch from input image using Maximal Poisson-disk Sampling (MPS)
%-------------------------------------------------------------------------------------

function [cut pos_flag pixel_x pixel_y] = imageMpsCrop(img, pos_flag, point_data, idx, new_width, new_height)
% INPUT:
% img: input image
% pos_flag: left upper, left lower, right upper, right lower
% point_data: mps sampling points, N*2
% idx: the index of sampling point
% new_width: the width of cropped image patch
% new_height: the height of cropped image patch
% OUTPUT:
% cut: cropped image patch
% pixel_x, pixel_y: the coordinate of the left upper of cropped patch in input image

[h, w, c] = size(img);
kCenterHeight = floor(new_height/2);
kCenterWidth = floor(new_width/2);
random_crop_flag = 0;
coor_x = point_data(idx, 1);
coor_y = 1 - point_data(idx, 2);
pixel_x = floor(coor_x * w);
pixel_y = floor(coor_y * h);
if pixel_x < 1
    pixel_x = 1;
end
if pixel_x > w
    pixel_x = w;
end
if pixel_y < 1
    pixel_y = 1;
end
if pixel_y > h
    pixel_y = h;
end

if pixel_x <= kCenterWidth
    pixel_x = 1;
    if pixel_y <= kCenterHeight
        % left upper;
        if pos_flag(1) 
            random_crop_flag = 1;
        else
            pixel_y = 1;
            pos_flag(1) = 1;
        end
    elseif pixel_y >= h - kCenterHeight
        % left lower;
        if pos_flag(2) 
            random_crop_flag = 1;
        else
            pixel_y = h - new_height + 1;
            pos_flag(2) = 1;
        end
    else
        pixel_y = pixel_y - kCenterHeight + 1;
    end
elseif pixel_x >= w - kCenterWidth
    pixel_x = w - new_width + 1;
    if pixel_y <= kCenterHeight
        % right upper;
        if pos_flag(3) 
            random_crop_flag = 1;
        else
            pixel_y = 1;
            pos_flag(3) = 1;
        end
    elseif pixel_y >= h - kCenterHeight
        % right lower;
        if pos_flag(4) 
            random_crop_flag = 1;
        else
            pixel_y = h - new_height + 1;
            pos_flag(4) = 1;
        end
    else
        pixel_y = pixel_y - kCenterHeight + 1;
    end
else
    pixel_x = pixel_x - kCenterWidth + 1;
    if pixel_y <= kCenterHeight
        pixel_y = 1;
    elseif pixel_y >= h - kCenterHeight
        pixel_y = h - new_height + 1;
    else
        pixel_y = pixel_y - kCenterHeight + 1;
    end
end

if random_crop_flag
    pixel_x = randi(w - new_width + 1);
    pixel_y = randi(h - new_height + 1);
end

if pixel_x < 1
    pixel_x = 1;
end
if pixel_x > w - new_width + 1
    pixel_x = w - new_width + 1;
end
if pixel_y < 1
    pixel_y = 1;
end
if pixel_y > h - new_height + 1
    pixel_y = h - new_height + 1;
end

cut = imcrop(img, [pixel_x, pixel_y, new_width - 1, new_height - 1]);
% if size(cut,1) ~= new_height || size(cut,2) ~= new_width
%     disp('error');
% end
end