function d=KLDivergence(x, y)
% This function compute KL Divergence
%
% ---Input---
% x: input value
% y: input value
% ---Output---
% d: KL divergence

d=x.*log(x./y)+(1-x).*log((1-x)./(1-y));

end