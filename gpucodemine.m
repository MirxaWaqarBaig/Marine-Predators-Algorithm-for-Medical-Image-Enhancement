function [denoised_img] = gpucodemine (input_img)
input_img = im2double(input_img);
input_img(isinf(input_img)|isnan(input_img)) = 0;
% Apply CLAHE
clahe_img = adapthisteq(input_img);
%%
% Apply Laplacian edge detection
laplacianFilter = [0 1 0; 1 -4 1; 0 1 0];
edge_img = imfilter(input_img, laplacianFilter, 'replicate');
%%
% Apply DN-CNN denoising
net = denoisingNetwork('dncnn');
%%
denoised_img = denoiseImage(input_img, net);
end