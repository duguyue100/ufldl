function g=sigmoidGradient(z)
% this function calculate gradient of sigmoid
%
% ---Input---
% z: input of the layer
% ---Output---
% g: gradient

g=sigmoidFunction(z).*(1-sigmoidFunction(z));

end