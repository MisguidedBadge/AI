%img_mat = table2array(test2Layer2);
T = dlmread('weights.dat', ';');
c = 1;
for i = 181:3:181
%     select = T(i, 1:9);
%     select = T(i, 1:30000);
%       select = T(i, 1:7500);
      select = T(i, 1:1000);
   % img_mat = table2array(select);
     plot(select);
     ylim([0 12]);
     ylabel('Error');
     xlabel('Epoch');
%     TS = reshape(select, [3 3]);
%       TS = reshape(select, [120 250]);
%      TS = reshape(select, [60 125]);
%      maximum = max(TS, [], 'all');
% % 
%      minimum = min(TS, [], 'all');
% % 
%      TS = 255 * (TS - minimum)/(maximum - minimum);
% % 
%      subplot(3,4,c);
%      imagesc(TS);
%      c = c + 1;
    %imshow(TS, [0 255]);
end