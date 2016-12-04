function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	auxiliaryTheta = [theta; -1];
	auxiliaryX = [X,y];
	
	% X * theta - y
	hypothesisSquareErrorVector = auxiliaryX * auxiliaryTheta;
	
	% x1Derivative = (X * theta - y) * x(i)
	%x1Derivative = hypothesisSquareErrorVector .* X(:,2);
	%x2Derivative = hypothesisSquareErrorVector .* X(:,3);
	
	
	for i = 1:size(theta)
		theta(i) = theta(i) - (1 / m) * alpha * sum(hypothesisSquareErrorVector .* X(:, i), 1)(1);
	end
	
	%theta = theta .- [(1 / m) * alpha * sum(hypothesisSquareErrorVector, 1)(1);
	%(1 / m) * alpha * sum(x1Derivative, 1)(1);
	%(1 / m) * alpha * sum(x2Derivative, 1)(1)];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
