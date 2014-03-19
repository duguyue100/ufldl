function [W, b, A]=initFeedforwardNetwork(noLayer, noNodes)
% This function returns necessary parameters of a feedforward network.
%
% ---Input---
% noLayer: number of layers.
% noNodes: number of nodes for each layer in N^{*}.
% ---Output---
% W: a cell with all initalized weights.
% b: a cell with all initalized bias.
% A: a cell with all layers' activations.

W=cell(noLayer-1, 1);
b=cell(noLayer-1, 1);
A=cell(noLayer, 1);

for i=2:noLayer
    [W{i-1}, b{i-1}]=initWeights(noNodes(i-1), noNodes(i));
end

end