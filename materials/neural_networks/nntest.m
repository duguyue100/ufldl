% Nerual Networks Tests
% Author: Hu Yuhuang (duguyue100)
% Date: 2014-03-17
% Email: duguyue100@gmail.com

%% init

clc;
clear;
close all;

%% Draw Sigmoid function

Z_sigmoid=-5:0.01:5;
Y_sigmoid=sigmoidFunction(Z_sigmoid);

figure, plot(Z_sigmoid, Y_sigmoid);
title('Sigmoid Function');
xlabel('z');
ylabel('f(z)');

%% Draw tanh function

Z_tanh=-5:0.01:5;
Y_tanh=tanhFunction(Z_tanh);

figure, plot(Z_tanh, Y_tanh);
title('tanh Function');
xlabel('z');
ylabel('f(z)');