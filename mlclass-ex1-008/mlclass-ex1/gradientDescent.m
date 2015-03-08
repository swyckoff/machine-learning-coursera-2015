function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

%    disp(computeCost(X,y,theta));

    h = X * theta;

%    disp(h(1:4,:));
%    fprintf('\nsize h: %f', size(h));

    err = h - y;
    
%    disp('\n err: ');
%    disp(err(1:4,:));
%    disp(size(err));

    gradient = X' * err;
    gradient_scaled = (alpha / m) * gradient;
    theta = theta - gradient_scaled;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
% disp('history');
% disp(J_history);
end
