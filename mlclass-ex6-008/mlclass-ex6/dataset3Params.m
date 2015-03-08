function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = Cs;

[v1, v2] = meshgrid(Cs,sigmas);
combos = [v1(:),v2(:)]

best_index = 1;
lowest_error = Inf;

predictions = zeros(size(combos));

for i=1:length(combos)
  C_guess = combos(i,1);
  sigma_guess = combos(i,2);

  fprintf(['Attempt #%d: C = %f, sigma = %f\n'],i,C_guess,sigma_guess);
  model= svmTrain(X, y, C_guess, @(x1, x2) gaussianKernel(x1, x2, sigma_guess));

  prediction = svmPredict(model, Xval);

  pred_error = mean(double(prediction ~= yval))
  
  if(pred_error < lowest_error)
    fprintf(['Lowest Error at #%d with %f'], i, pred_error);

    best_index = i;
    lowest_error = pred_error;
  end
end

C = combos(best_index,1);
sigma = combos(best_index,2);

fprintf(['Picking C = %f, sigma = %f'],C,sigma);


% =========================================================================

end
