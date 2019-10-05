a = csvread('test_2Layer.dat',0,0);
%b = csvread('test_1Layer.dat',0,0);
c  = a(:,1)';
t = 1:1:300;

subplot(2,3,1);
hold on
plot(t,(a(:,1))', '-blue');
%plot(t,(b(:,1))', '-red');

subplot(2,3,2);
hold on;
plot(t,(a(:,2))', '-blue');
%plot(t,(b(:,2))', '-red');

subplot(2,3,3);
hold on;
plot(t,(a(:,3))', '-blue');
%plot(t,(b(:,3))', '-red');

subplot(2,3,4);
hold on;
plot(t,(a(:,4))', '-blue');
%plot(t,(b(:,4))', '-red');

subplot(2,3,5);
hold on;
plot(t,(a(:,5))', '-blue');
%plot(t,(b(:,5))', '-red');

