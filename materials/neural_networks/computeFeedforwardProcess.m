function A=computeFeedforwardProcess(X, W, b, A, noLayer)
% this function perform feedforward pass for all layers
%
% ---Input---
% X: input data
% W: weights
% b: bias
% noLayer: total number of layers
% ---Output---
% A: computed activations

A{1}=X;
for i=2:noLayer
    A{i}=calculateActivation(A{i-1}, W{i-1}, b{i-1});
end

end