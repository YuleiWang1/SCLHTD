clear
clc
load sandiego
load groundtruth
[H, W, D] = size(data);
a = reshape(data,[H*W,D]);
b = a';
agumentation_one = b(1:2:188, :);
agumentation_two = b(2:2:189, :);
agumentation_one = agumentation_one';
agumentation_two = agumentation_two';
agumentation_one = reshape(agumentation_one, [120,120,94]);
agumentation_two = reshape(agumentation_two, [120,120,94]);
save('agumentation_one');
save('agumentation_two');
c = find(gt == 1);
target = a(c,:);
target = mean(target);
target_one = target(1:2:188);
target_two = target(2:2:189);
save('target_one');
save('target_two');
save('target');


%figure(1)
%plot(target_one);
%figure(2)
%plot(target_two);
%ooo = reshape(agumentation_one(:,94),[100,100]);
%ppp = reshape(agumentation_two(:,94),[100,100]);
%figure(1)
%imagesc(ooo)
%figure(2)
%imagesc(ppp)