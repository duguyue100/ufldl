function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
%cost = 0;

%thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% compute hypothesis

M=size(data,2);
h=exp(theta*data);
h_sum=1./sum(h,1);
h=h.*repmat(h_sum, numClasses, 1);

% compute elements in cost function.
logh=log(h);

% cost

cost=(-1/M)*sum(sum(groundTruth.*logh))+(lambda/2)*sum(sum(theta.*theta));

thetagrad = (-1/M)*((groundTruth-h)*data')+lambda*theta;




% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

