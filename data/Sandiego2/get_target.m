clear
clc

load sandiego
load groundtruth
a = find(gt==1);
[H,W,D] = size(data);
data = reshape(data,[H*W,D]);
target = data(a,:);
prior_target = target(52,:)';
save('prior_target')