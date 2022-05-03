clc,clear;
theta = [1.66085011783622	0.742849382342569
1.21182638400520	1.33755284373052
0.934863518802253	-0.115239433007944
-0.544254020126906	-0.502432353057892
1.06463008366569	0.670601834011999
0.197241340223863	-0.240893152284759
0.213126039525263	-0.0648620768099279
];

data = readtable('dataset.xlsx');
X = table2array(data(:, 4:9));
Xmean = mean(X, 1);
Xvar = var(X, 1);

testdata = readtable('dataset.xlsx');
Xtest = table2array(testdata(:, 4:9));
Xtest = (Xtest - Xmean) ./ Xvar;
[mtest, ntest] = size(Xtest);

Xtest = [ones(mtest, 1) Xtest];
[ptest, prob] = predict(theta, Xtest);

function [ypred, proboffire] = predict(W, X)
    [m, ~] = size(X);
    F = X * W;
    p = exp(F) ./ sum(exp(F), 2);
    ypred = zeros(m, 1);
    prob = zeros(m, 1);
    proboffire = zeros(m, 1);
    for i = 1:m
        prob(i) = max(p(i, :));
        ypred(i) = find(p(i, :) == prob(i)) - 1;
        if ypred(i) == 1
            proboffire(i) = prob(i);
        else
            proboffire(i) = 1 - prob(i);
        end
    end
end