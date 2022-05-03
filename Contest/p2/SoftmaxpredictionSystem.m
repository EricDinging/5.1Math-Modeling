clear, clc;
%logistic regression F1 softmax
data = readtable('dataset.xlsx');
testdata = readtable('testdataset.xlsx');
X = table2array(data(:, 4:9));
y = table2array(data(:, 10));
Xtest = table2array(testdata(:, 4:9));
%data normalization
Xmean = mean(X, 1);
Xvar = var(X, 1);
X = (X - Xmean) ./ Xvar;
Xtest = (Xtest - Xmean) ./ Xvar;
%data visualization;


ptest = 0;
[m, n] = size(X);
X = [ones(m, 1), X];
mtrain = round(m * 0.7);
[mtest, ntest] = size(Xtest);
Xtest = [ones(mtest, 1) Xtest];
best_f1 = 0;
while (sum(ptest) == 0 || sum(ptest) > 5 || best_f1 < 0.3 )
    %data splitting
    randind = randperm(m);
    Xtrain = X(randind(1:mtrain), :); ytrain = y(randind(1:mtrain));
    Xval = X(randind(mtrain+1:m), :); yval = y(randind(mtrain+1:m));
    pd = makedist('Normal');
    initial_theta = random(pd, [n + 1, 2]);
    best_lambda = 0; best_theta = initial_theta;
    for lambda = 0:0.5:10
        options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
        [theta, cost] = fminunc(@(t)(softmaxcostReg(t, Xtrain, ytrain, lambda)), initial_theta, options);
        [pval, ~] = predict(theta, Xval);

        f1 = f1score(pval, yval);

        if f1 > best_f1
            best_f1 = f1;
            best_lambda = lambda;
            best_theta = theta;
        end
    end
    
    [ptest, prob] = predict(best_theta, Xtest);
end

[ptrain, ~] = predict(best_theta, Xtrain);
[pval,~] = predict(best_theta, Xval);
fprintf('Best lambda: %f\n', best_lambda);
fprintf('Train F1 Score: %f\n', f1score(ptrain, ytrain));
fprintf('Train Accuracy: %f\n', mean(double(ptrain == ytrain)))
fprintf('Validation F1 Score: %f\n', best_f1);
fprintf('Validation Accuracy: %f\n', mean(double(pval == yval)));
%predicting excel 3

fprintf('1 dataset number: %f\n', length(find(y == 1)));
fprintf('0 dataset number: %f\n', length(find(y == 0)));
fprintf('1 training prediction number: %f\n', length(find(ptrain==1)));
fprintf('1 validation number: %f\n', length(find(yval == 1)));
fprintf('1 validation prediction number: %f\n', length(find(pval == 1)));
fprintf('1 testing prediction number: %f\n', length(find(ptest == 1)));



% data visualization, decision boundary






function [loss, dW] = softmaxcostReg(W, X, y, reg)
    [m, ~] = size(X);
    F = X * W;
    Fn = F - max(F, 2);
    p = zeros(m, 1);
    Pd = exp(Fn) ./ sum(exp(Fn), 2);
    for i = 1:m
        p(i) = Pd(i, y(i) + 1);
        Pd(i, y(i) + 1) = Pd(i, y(i) + 1) - 1;
    end
    logp = log(p);
    loss = - sum(logp) / m + reg * sum(W.^2, 'all');
    dW = X' * Pd;
    dW = dW / m;
    dW = dW + 2 * reg * W;
end

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

function f1 = f1score(p, y)
    tp = sum(p == 1 & y == 1);
    fp = sum(p == 1 & y == 0);
    fn = sum(p == 0 & y == 1);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * (precision * recall)/(precision + recall);
end
