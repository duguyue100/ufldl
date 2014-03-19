% Backpropagation Algorithm Tests
% Author: Hu Yuhuang (duguyue100)
% Date: 2014-03-17
% Email: duguyue100@gmail.com

%% init

clc;
clear;
close all;

%% Backpropagation Algorithm Demonstration

noLayer=3;
noNodes=[2,2,1];
[W, b, A]=initFeedforwardNetwork(noLayer, noNodes);

% simulate XOR operation
X=[0,0,1,1;
   0,1,0,1];
Y=[0,1,1,0];

% calculate feedforward process;
for i=1:30
    A=computeFeedforwardProcess(X, W, b, A, noLayer);
    [W, b]=computeBackPropagation(A, W, b, Y, noLayer, 0.2, 1);
    A{3}
end

