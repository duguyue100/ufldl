function next_A=calculateActivation(pre_A, W, b)
% This function calcualte activation for next layer based on previous
% layer's activation, weights and bias in feedforward manner.
%
% ---Input---
% pre_A: previous layer's activation in R^{n}
% W: Weights between previous layer and next layer in R^{m*n} (m is number
% of next layer's nodes)
% b: Bias of previous layer in R^{n}

z=W*pre_A+b;

next_A=sigmoidFunction(z);

end