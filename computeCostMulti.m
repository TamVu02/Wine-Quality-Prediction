function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

predict = X*theta;
J = 1/(2*m) * sum((predict - y).^2);

end
