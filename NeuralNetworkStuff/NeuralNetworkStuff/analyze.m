%img_mat = table2array(test2Layer2);
T = readtable('weights.dat');

for i = 1:24
    select = T(i, 1:7500);

    img_mat = table2array(select);

    TS = reshape(img_mat, [125 60]);


    maximum = max(TS, [], 'all');

    minimum = min(TS, [], 'all');

    TS = 255 * (TS - minimum)/(maximum - minimum);

    subplot(2,12,i);
    imagesc(TS);
    %imshow(TS, [0 255]);
end