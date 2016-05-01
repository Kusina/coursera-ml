% Exercise 2: Regularized Logistic Regression

clear; close all; clc

data = load('ex2data2.txt');
X = data(:, 1:2);
y = data(:, 3);

plot_data(X, y);

hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

%% Regularization

% Add polynomial features
X = map_feature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

[cost, grad] = cost_function_reg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%% Regularization and Accuracies

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter
% lambda = 0;
lambda = 1;
% lambda = 10;
% lambda = 100;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(cost_function_reg(t, X, y, lambda)), initial_theta, options);

plot_decision_boundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);