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

hx = X*theta

% goal to ignore the first param of theta, since it's theta_0.  we don't regularize that.
trick_theta = theta(2:size(theta));
trick_theta = [0;trick_theta];

J = trick_theta'*trick_theta;

J = 1/(2*m) * sum((hx - y).^2) + lambda/(2*m) * trick_theta'*trick_theta


grad = (1/m) * (X'* (hx - y) + lambda*trick_theta);


% =========================================================================

grad = grad(:);

end
