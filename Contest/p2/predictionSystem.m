clear, clc;
%logistic regression F1
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
while (sum(ptest) == 0 || sum(ptest) > 5 )
        %data splitting
    [m, n] = size(X);
    X = [ones(m, 1) X];
    % index = find(y == 1);
    % X = [X; X(index,:); X(index,:);X(index,:)];
    % y = [y; y(index); y(index); y(index)];
    [m, ~] = size(X);
    mtrain = round(m * 0.7);
    randind = randperm(m);
    Xtrain = X(randind(1:mtrain), :); ytrain = y(randind(1:mtrain));

    % index = find(ytrain == 1);% oversampling
    % Xtrain = [Xtrain; Xtrain(index, :); Xtrain(index, :); Xtrain(index, :); Xtrain(index, :)];
    % ytrain = [ytrain; ytrain(index); ytrain(index); ytrain(index);ytrain(index)];

    Xval = X(randind(mtrain+1:m), :); yval = y(randind(mtrain+1:m));
    pd = makedist('Normal');
    initial_theta = 0.01 * random(pd, [n + 1, 1]);


    [mtest, ntest] = size(Xtest);
    Xtest = [ones(mtest, 1) Xtest];

    best_f1 = 0; best_lambda = 0; best_threshod = 0;
    for lambda = [0, 0.01 , 0.1, 0.2, 0.5, 1, 2, 5, 10]
        for threshod = 0.1:0.05:0.6
            options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
            [theta, cost] = fminunc(@(t)(costFunctionReg(t, Xtrain, ytrain, lambda)), initial_theta, options);
            pval = predict(theta, Xval, threshod);

            f1 = f1score(pval, yval);

            if f1 > best_f1
                best_f1 = f1;
                best_lambda = lambda;
                best_threshod = threshod;
                best_theta = theta;
            end
         end
    end
    
    ptest = predict(best_theta, Xtest, best_threshod);
end

ptrain = predict(best_theta, Xtrain, best_threshod);
pval = predict(best_theta, Xval, best_threshod);
prob = sigmoid(Xtest * best_theta) * 100;
fprintf('Best lambda: %f\n', best_lambda);
fprintf('Best threshod: %f\n', best_threshod);
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


function g = sigmoid(z)
    g = 1./(1+exp(-z));
end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);  
    J = 0;
    grad = zeros(size(theta));
    ntheta = theta;
    ntheta(1) = 0;
    J = -1/m * y' * log(sigmoid(X * theta)) - 1/m * (1-y)'* log(1 - sigmoid(X * theta)) + lambda / (2*m) * sum(ntheta.^2);
    grad = 1/m * X' * (sigmoid(X * theta) - y) + lambda / m * ntheta;
end

function p = predict(theta, X, threshod)
    m = size(X, 1);
    p = zeros(m, 1);
    p = sigmoid(X * theta);
    p = p > threshod;
end

function f1 = f1score(p, y)
    tp = sum(p == 1 & y == 1);
    fp = sum(p == 1 & y == 0);
    fn = sum(p == 0 & y == 1);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * (precision * recall)/(precision + recall);
end
