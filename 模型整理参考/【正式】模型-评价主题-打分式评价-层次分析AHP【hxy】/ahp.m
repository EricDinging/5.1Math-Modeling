function [weight, CR] = ahp(Matrix)
%% Description
%{ 
Function: Use AHP to calculate weight vector.
Usage: [weight, CR]=ahp(Matrix)
Input: 
    Mitrix - discriminant matrix of n indicators
Output:
    weight - weight vector of n indicators
    CR - consistency ratio
Example: 
    Matrix = [1 2 4; 0.5 1 2; 0.25 0.5 1]
    [weight, CI] = ahp(Matrix)
%}
%% Implementation
% Calculate the number n of samples
[number,~] = size(Matrix);
% Get RI(n)
RI = [0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51];
ri = RI(number);
% Calculate the eigenvalue V and eigenvector U
[U,V] = eig(Matrix);
% Find the maximum eigenvalue Max_eig
Max_eig = max(max(V));
% Find the eigenvector c of Max_eig
[~,c] = find(V == Max_eig,1);
% Calculate weight vector
weight = U(:,c) ./ sum(U(:,c));
%% Consistency test 
% Calculate consistency indicator(CI)
CI = (Max_eig - number)/(number - 1);
% Calculate consistency ratio(CR)
CR = CI/ri;
% Result of the consistency test
if CR < 0.10
    disp('Since CR<0.10, so the consistency is acceptable!');
else
    disp('Note: CR>=0.10, so the judgment matrix A needs to be modified!');
end