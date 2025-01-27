clc, clear
% Input data with rows of samples and columns of indexes
a = [0.71	0.49	0.41	0.51	0.46
0.40	0.49	0.44	0.57	0.50
0.55	0.56	0.48	0.53	0.49
0.62	0.93	0.38	0.53	0.47
0.45	0.42	0.41	0.54	0.47
0.36	0.37	0.46	0.54	0.48
0.55	0.68	0.42	0.54	0.46
0.62	0.90	0.38	0.56	0.46
0.61	0.99	0.33	0.57	0.43
0.71	0.93	0.35	0.66	0.44
0.59	0.69	0.36	0.57	0.48
0.41	0.47	0.40	0.54	0.48
0.26	0.29	0.43	0.57	0.48
0.14	0.16	0.43	0.55	0.47
0.12	0.13	0.45	0.59	0.54
0.22	0.25	0.44	0.58	0.52
0.71	0.49	0.41	0.51	0.46]; 
% Standardize data
standardized_a = zscore(a);
% Calculate corrcoef matrix
r = corrcoef(standardized_a);
% Calculate eigenvalues y, eigenvectors x, contribution p
[x, y, p] = pcacov(r);
% Construct row vector of +1/-1
f = sign(sum(x));
% Modify the sign of eigenvectors x
x = x .* f;
% Choose the number of principle components
num = 3;
% Calculate the score of each principles
new_y = standardized_a * x(:,(1:num));
% Calculate the comprehensive score z
z = new_y * p(1:num) / 100;
% Sort the score from largest to smallest (index = 1 means year = 1984)
[sorted_z, index] = sort(z,'descend');