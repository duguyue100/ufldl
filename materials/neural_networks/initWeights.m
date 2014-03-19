function [W, b]=initWeights(n, m)
% this function generate a set of predefined random weights and bias
%
% ---Input---
% m: number of next layer's nodes
% n: number of previous layer's nodes
% epsilon: standard deviation of a zero mean Gaussian random process.
% ---Output---
% W: weights with random values
% b: bias of previous layer

epsilon_init = 0.12;
Wtemp = rand(m,1+n)*2*epsilon_init-epsilon_init;

W=Wtemp(:,1:n);
b=Wtemp(:,n+1);

end