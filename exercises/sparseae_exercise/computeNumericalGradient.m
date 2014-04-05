function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON=0.0001;
L=size(theta, 1);
E=eye(L)*EPSILON;

%Tp=repmat(theta, 1, size(theta, 1))+E;
%Tm=repmat(theta, 1, size(theta, 1))-E;
%Tcp=num2cell(Tp,1);
%Tcm=num2cell(Tm,1);

%numgrad1 = cellfun(@(c) J(c(1:size(E,1))), Tcp);
%numgrad2 = cellfun(@(c) J(c(1:size(E,1))), Tcm);

%numgrad=transpose((numgrad1-numgrad2)./(2*EPSILON));

for i=1:size(theta,1)
    numgrad(i)=(J(theta+E(:, i))-J(theta-E(:, i)))./(2*EPSILON);
    disp(i)
end




%% ---------------------------------------------------------------
end