% This file contains tests on Autoencoders and Sparsity
% Author: Hu Yuhuang
% Date: 2014-03-30

%% init

clc;
clear;
close all;

%% plot KL divergence

X=0:0.001:1;
rho=0.2;

Y=KLDivergence(X, rho);

figure,
plot(X,Y);
xlabel('average activation on hidden unit');
ylabel('KL divergence');