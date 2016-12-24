function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
regTheta = theta;
regTheta(1,:) = 0;
J = (1/(2 * m)) * (sumsq(X*theta - y, 1)) + sumsq(regTheta) * lambda / 2 / m;

%grad = (X * theta - y)' * X;

auxiliaryTheta = [theta; -1];
auxiliaryX = [X,y];

% X * theta - y
hypothesisSquareErrorVector = auxiliaryX * auxiliaryTheta;

% auxiliarySumHypothesisDerivative = (X * theta - y) * x(i)
auxiliarySumHypothesisDerivative = hypothesisSquareErrorVector .* X;

% sum each column of matrix to get gradient vector
grad = sum(auxiliarySumHypothesisDerivative, 1) ./ m;

% regularize gradient
grad = grad + (lambda .* regTheta' ./ m);
% =========================================================================

grad = grad(:);

end
