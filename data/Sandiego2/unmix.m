clear
clc
load sandiego
load target
[H, W, D] = size(data);
data = reshape(data,[H*W, D])';
[U, indices] = hyperAtgp(data, 30, target');


[lenth, bands]=size(U);
tgt = target';
dis=zeros(1,bands);
for is=1:bands
    xx=U(:,is);
    xx=xx(:);
    dis(is)=(dot(xx,tgt)/(norm(xx)*norm(tgt)));
end
sorce = find(dis >= 0.98);
prior = U(:,sorce);
%prior_target = mean(prior,2);
prior_target = prior(:, 4);
save('prior_target');


