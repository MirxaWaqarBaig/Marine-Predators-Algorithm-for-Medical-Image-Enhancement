clear all
clc
%%
input_img = im2double(dicomread('Subject_13.dcm'));
%%
tech_21 = mpaenhancement(input_img);
%%
figure()
imshow(tech_21,[]);