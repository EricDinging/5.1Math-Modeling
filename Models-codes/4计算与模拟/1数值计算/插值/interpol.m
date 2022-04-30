x0 = [0 3 5 7 9 11 12 13 14 15];
y0 = [0 1.2 1.7 2.0 2.1 2.0 1.8 1.2 1.0 1.6];
x = 0:0.1:15;
y1 = interp1(x0, y0, x);
% plot(x0, y0, '+', x, y1);
y2 = interp1(x0, y0, x, 'spline');
plot(x0, y0, '+', x, y2);

%%
clear, clc
x = 100:100:500;
y = 100:100:400;
z = [636 697 624 478 450
    698 712 630 478 420
    680 674 598 412 400
    662 626 552 334 310];
pp = csape({x, y}, z');
xi = 100:10:500;
yi = 100:10:400;
cz = fnval(pp, {xi, yi});
mesh(cz);