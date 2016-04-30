%% Exercise 1: Linear Regression

% ex1data1.txt - dataset for univariate linear regression
% ex1data2.txt - dataset for multivariate linear regression

clear; close all; clc

%% Plotting

data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); % Number of training examples

plot_data(X, y);

%% Gradient descent

X = [ones(m,1), X]; % Take into account the intercept term (theta_0)
theta = zeros(2, 1); % Initialize fitting parameters

% Gradient descent settings
iterations = 1500;
alpha = 0.01;

% Cost function
compute_cost(X, y, theta)

% Run gradient descent
theta = gradient_descent(X, y, theta, alpha, iterations);

fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

hold on;
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% Visualizing J(theta_0, theta_1)

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% Initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));


% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = compute_cost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
