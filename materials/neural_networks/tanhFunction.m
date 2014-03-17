function y=tanhFunction(z)
% This function implements one activation function -- tanh function
% y=f(z)=(exp(z)-exp(-z))/(exp(z)+exp(-z)) where y, z in R.
%
% ---Input---
% z: the input of sigmoid function.
% ---Output---
% y: the output of sigmoid function.

y=(exp(z)-exp(-1*z))./(exp(z)+exp(-1*z));

end