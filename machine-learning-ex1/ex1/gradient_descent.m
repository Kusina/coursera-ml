function [theta, J] = gradient_descent(X, y, theta, alpha, num_iters)
%GRADIENT_DESCENT Performs gradient descent to learn theta
%   theta = GRADIENT_DESENT(X, y, theta, alpha, num_iters) updates
%   theta by taking num_iters gradient steps with learning rate alpha

m = length(y);
J = zeros(num_iters, 1);

for i = 1:num_iters
    err = (X*theta - y)';
    theta_0 = theta(1) - (alpha/m)*err*X(:,1);
    theta_1 = theta(2) - (alpha/m)*err*X(:,2);
    theta(1) = theta_0;
    theta(2) = theta_1;
    
    J(i) = compute_cost(X, y, theta);
end

end

