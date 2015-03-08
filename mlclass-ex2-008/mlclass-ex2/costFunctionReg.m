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

z = X * theta;

hx = sigmoid(z);

% This is to avoid theta_0 being regularized
trick_theta = theta(2:size(theta));
trick_theta = [0;trick_theta];

J = (-1/m)*(y'*log(hx)+(1-y)'*log(1-hx)) + (lambda/(2*m))*trick_theta'*trick_theta;


% X(:,i) outputs the ith column, since : does every row.
% hx is the sigmoid function for all values in z where z=X * theta.
% X * theta gives us X parameterized by theta
% sum((hx - y).*X(:,i)) gives us the sum of the...
%   error: hypothesis - y (the actual dependent values) 
%   times each element of our training data X 

  grad = (1/m) * (X'* (hx - y) + lambda*trick_theta); 

% =============================================================

end
