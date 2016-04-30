function [theta, J_history] = gradient_descent_multi(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates
%   theta by taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);

for i = 1:num_iters
    err = (X*theta - y)';
    theta = theta - (alpha/m)*(err*X)';
    J_history(i) = compute_cost(X, y, theta);
end

end

