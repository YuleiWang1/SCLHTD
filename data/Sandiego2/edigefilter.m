clear
clc

load('D:\Workspace\contrastive_target_detection\result\Sandiego2.mat')
load firstpca

fo = firstpca;
sigma_s = 5;
sigma_r = 2; % 0.5 sandiego2=2
result = RF(double(detect), sigma_s, sigma_r, 3, fo);
save('result');
