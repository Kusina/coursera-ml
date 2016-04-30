function J = compute_cost(X, y, theta)
%COMPUTE_COST Compute cost for linear regression
%   J = COMPUTE_COST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);
J = (1/(2*m)) * (X*theta - y)'*(X*theta - y);

end

