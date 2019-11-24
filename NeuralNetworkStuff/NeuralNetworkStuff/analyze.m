%img_mat = table2array(test2Layer2);
%T = readtable('test_2layer.dat');

select = T(39, 1:30000);

img_mat = table2array(select);

TS = reshape(img_mat, [250 120]);


maximum = max(TS, [], 'all');

minimum = min(TS, [], 'all');

TS = 255 * (TS - minimum)/(maximum - minimum);

imshow(TS, [0 255]);