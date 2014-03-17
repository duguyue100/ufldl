function [W, b]=initWeights(n, m)
% this function generate a set of predefined random weights and bias
%
% ---Input---
% m: number of next layer's nodes
% n: number of previous layer's nodes
% ---Output---
% W: weights with random values
% b: bias of previous layer

W=rand(m,n);
b=rand(n,1);

end