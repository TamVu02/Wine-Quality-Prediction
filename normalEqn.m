function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);

theta=((transpose(X)*X)^(-1))*transpose(X)*y;

end
