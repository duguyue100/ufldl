function [newW, newb]=computeBackPropagation(A, W, b, Y, noLayer, alpha, lambda)
% this function updates weights and bias based on A
%
% ---Input---
% A: computed output
% W: weights
% b: bias
% Y: actual output
% noLayer: total number of layers
% alpha: learning rate
% lambda: weight decay parameter
% ---Output---
% newW: updated weights
% newb: updated bias

%% paramters

M=size(Y, 2); % number of samples

%% calculate error for each layer

deltaW=cell(size(W));
deltab=cell(size(b));
delta=cell(size(A));

delta{noLayer}=-1*(Y-A{noLayer}).*sigmoidGradient(calculateZ(A{noLayer-1}, W{noLayer-1}, b{noLayer-1}));

for i=noLayer-1:2
    delta{i}=(transpose(W{i})*delta{i+1}).*sigmoidGradient(calculateZ(A{i-1}, W{i-1}, b{i-1}));
end

for i=1:noLayer-1
    deltaW{i}=delta{i+1}*transpose(A{i});
    deltab{i}=sum(delta{i+1}, 2);
end

%% update parameters

newW=W;
newb=b;
for i=1:noLayer-1
    newW{i}=newW{i}+alpha*((1/M)*deltaW{i}+lambda*W{i});
    newb{i}=newb{i}+alpha*((1/M)*deltab{i});
end

end