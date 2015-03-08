function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

z = X * theta;

hx = sigmoid(z);

J = (-1/m)*(y'*log(hx)+(1-y)'*log(1-hx))


% X(:,i) outputs the ith column, since : does every row.
% hx is the sigmoid function for all values in z where z=X * theta.
% X * theta gives us X parameterized by theta
% sum((hx - y).*X(:,i)) gives us the sum of the...
%   error: hypothesis - y (the actual dependent values) 
%   times each element of our training data X 
for i=1:length(theta)
  grad(i) = (1/m) * sum((hx - y).*X(:,i)) 
end

% =============================================================

end
