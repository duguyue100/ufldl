function y=sigmoidFunction(z)
% This function implements one activation function -- sigmoid function
% y=f(z)=1/(1+exp(-z)) where y, z in R.
%
% ---Input---
% z: the input of sigmoid function.
% ---Output---
% y: the output of sigmoid function.

y=1./(1+exp(-1*z));

end