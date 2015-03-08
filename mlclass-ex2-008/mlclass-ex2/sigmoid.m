function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%sizeor10 = min(length(z), 10);

%disp(z(1:sizeor10,:));

% Okay let's be imperative... boo.

% sig_func = 1 / ( 1 + e^(-z));

for i=1:rows(z)
  for j = 1:columns(z)
      g(i,j) = 1 / ( 1 + e^(-z(i,j)));
  end
end
% g = arrayfun(@(x) sig_func(x),g); % almost works...

% =============================================================

end
