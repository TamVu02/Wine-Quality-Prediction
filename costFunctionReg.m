function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h0=sigmoid(X*theta);
J=(1/m)*sum(-y.*log(h0)-(1-y).*log(1-h0))+((lambda/(2*m))*sum(theta(2:end).^2));
grad=(1/m)*sum((h0-y).*X);
grad(2:end)=grad(2:end)+(lambda/m).*theta(2:end)';


end
