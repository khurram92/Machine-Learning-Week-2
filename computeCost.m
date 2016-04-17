function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%J = (0.5/length(y)) * sum(((theta' * X')' - y).^2);
% X:m x 2
% theta: 2 * 1
% Product: m x 1

m = length(y); % sample size
J = sum(((X * theta) - y).^2)/(2 * m); % cost function


% =========================================================================

end
