function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

regTheta = theta;
regTheta(1) = 0;
J = (sum(- log(sigmoid(X * theta)) .* y - (1 .- y) .* log(1 .- sigmoid(X * theta)) , 1) / m) + lambda / (2 * m) * sum(regTheta .^ 2);


% ****************** Vectorization *****************************************
grad = (sigmoid(X * theta) - y)' * X;
grad = (1 / m) .* grad + lambda * regTheta' / m;


% **************** Previous implementation ********************************************
%for j = 1:size(theta)
%	if (j != 1)
%		grad(j) = (1 / m) * sum((sigmoid(X * theta) - y) .* X(:, j), 1)(1) + lambda / m * theta(j);
%	else
%		grad(j) = (1 / m) * sum((sigmoid(X * theta) - y) .* X(:, j), 1)(1);
%	end
%end

%grad = (1 / m) .* ((sigmoid(X * theta) - y)' * X);

%for j = 1:size(theta)
%	if (j != 1)
%		grad(j) = grad(j) + lambda / m * theta(j);
%	else
%		grad(j) = grad(j);
%	end
%end
% ****************** End previous implementation ********************************


% =============================================================

end